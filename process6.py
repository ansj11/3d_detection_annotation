import os
import cv2
import json
import argparse
# import threading
# import subprocess
import numpy as np
from tqdm import tqdm
# from multiprocessing import Process, Lock
from pdb import set_trace
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import optim
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
from gaussian_object.gaussian_render import render as GS_Renderer
from gaussian_object.cameras import Camera as GS_Camera
from gaussian_object.utils.graphics_utils import focal2fov
from gaussian_object.gaussian_model import GaussianModel
import config as CFG
from pytorch_msssim import SSIM, MS_SSIM
from gaussian_2dgs.arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision.transforms as transforms

# 2dgs setting 
from gaussian_2dgs.gaussian_model import GaussianModel as GaussianModel2D
from gaussian_2dgs.gaussian_renderer import render as GS_Renderer2D
from gaussian_2dgs.scene.cameras import Camera as GS_Camera2D

import  kornia.feature as KF
import kornia as K
from kornia_moons.viz import draw_LAF_matches
# from distort import ImageDistortionModel

matcher = KF.LoFTR(pretrained='outdoor').cuda()
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

device = torch.device('cuda')

depth_pipe = None
# depth_pipe = DepthAnythingV2(**model_configs['vitl'])
# depth_pipe.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
# depth_pipe = depth_pipe.to(device).eval()

parser = argparse.ArgumentParser()
###### arguments for 3D-Gaussian Splatting Refiner ########
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP  = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
parser.add_argument("--input", type=str, default='',
                        help="input path")
parser.add_argument("--fx", type=str, default=3630,
                        help="input path")
parser.add_argument("--fy", type=str, default=3600,
                        help="input path")
args = parser.parse_args()

MSELoss = torch.nn.MSELoss(reduction='none')
L1Loss = torch.nn.L1Loss(reduction='none')
SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
# MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)
# SSILoss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)


def get_mask2(mask, xyxy):
    x0, y0, x1, y1 = xyxy
    indices = np.unique(mask[y0:y1, x0:x1])
    max_area = 0
    max_idx = -1
    for idx in indices:
        if idx == 0:
            continue
        ys, xs = np.where(mask == idx)
        x2, y2 = xs.min(), ys.min()
        x3, y3 = xs.max(), ys.max()
        s1 = (x1-x0)*(y1-y0)
        s2 = (x3-x2)*(y3-y2)
        xmin = np.maximum(x0, x2)  # 左上角的横坐标
        ymin = np.maximum(y0, y2)  # 左上角的纵坐标
        xmax = np.minimum(x1, x3)  # 右下角的横坐标
        ymax = np.minimum(y1, y3)  # 右下角的纵坐标
        overlap = max(0, xmax - xmin) * max(0, ymax - ymin)
        iou = overlap / (s1 + s2 - overlap)
        if iou > max_area and iou > 0.2:
            max_area = iou
            max_idx = idx

    return (mask == max_idx).astype(np.uint8) * 255, max_idx


"""process all objects without occlusion"""
def main(mode='bbox', debug=True, padding_pixel=0):
    root = "/gemini/data-2/segment/"

    path = args.input
    with open(path, 'r') as f:
        track_dict = json.load(f)

    img_path = path.replace("dicts", "trial_v2")[:-4] + 'jpg'

    json_path = img_path.replace('trial_v2', 'poses_v3')[:-3] + 'json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    image = cv2.imread(img_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    mask_path = img_path.replace('trial_v2', 'masks').replace('jpg', 'png')
    mask = cv2.imread(mask_path, 0)
    
    fx, fy = args.fx, args.fy
    cx, cy = 1344.0, 760.0
    device = torch.device('cuda')
    
    dic = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            dic = json.load(f)
    for track_id, result in track_dict.items():
        bbox = result['bbox']
        brand, subbrand, year, color = result['brand']
        # kpts2d = np.array(result['kpts'])
        if max(bbox[-2:]) <= 40:
            print(bbox[-2:], " <= 50", result['brand'])
            continue
        if "一汽" not in brand:
            continue
        xyxy = result['xyxy']
        x0, y0, x1, y1 = xyxy
        crop = image[y0:y1, x0:x1]
        rgba = np.zeros((height, width, 4), dtype='uint8')
        rgba[y0:y1, x0:x1, :3] = crop
        alpha, max_idx = get_mask2(mask, xyxy)
        rgba[..., -1] = alpha
        
        if not isinside(alpha) or alpha.max() == 0:
            print('not inside or no mask', result['brand'], alpha.max())
            continue

        occlude = result['occlude']
        occlude_mask = np.zeros_like(alpha)
        dilate = cv2.dilate(alpha[y0:y1, x0:x1], np.ones((3,3), np.uint8), iterations=2)
        indices = np.unique(mask[y0:y1, x0:x1][dilate > 0])

        if len(indices) > 2:
            print("is occluded mask", result['brand'])
            for idx in indices[1:]:
                if idx == max_idx: continue
                occlude_mask[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] == idx).astype(np.uint8) * 255
            occlude = True
        
        FovX, FovY = result['fovxy']
        queries = result['queries']
        for query in queries:
            gs_dir = query['gs_path']

            gs_path = os.path.join(gs_dir, 'point_cloud/iteration_7000/point_cloud.ply')
            if not os.path.exists(gs_path):
                print("not exists...", gs_dir, result['brand'])
                continue

            # kpts3d = np.array(queries['kpts3d'])
            
            pH, pW = height, width
            if padding_pixel > 0:
                padding_pixel = int(min(padding_pixel, max(height, width) * 0.2))
                print('padding pixel: ', padding_pixel)
                left = int(round(width / 2 + padding_pixel - cx))
                right = int(padding_pixel * 2 - left)
                top = int(round(height / 2 + padding_pixel - cy))
                down = int(padding_pixel * 2 - top)  # wrap affine左移，则cx右移
                pad_image = np.zeros((height+padding_pixel*2, width+padding_pixel*2, 4), dtype='uint8')
                pad_image[top:-down, left:-right] = rgba
                rgba = pad_image
                cv2.imwrite('ansj/%s.jpg' % brand, image)
                print(left, right, top, down)
                pH, pW = rgba.shape[:2]
                cx += left
                cy += top
                bbox[0] += left
                bbox[1] += top
                # kpts2d[:,0] += left
                # kpts2d[:,1] += top
                xyxy = [xyxy[0] + left, xyxy[1] + top, xyxy[2] + left, xyxy[3] + top]
            
            gaussians = GaussianModel2D(3)
            gaussians.load_ply([gs_path])
            rgba_tensor = torch.from_numpy(rgba/255.0).to(device).permute(2, 0, 1)
            mask_tensor = rgba_tensor[-1:].float()
            image_tensor = rgba_tensor[:-1].float() * mask_tensor
            omask_tensor = torch.from_numpy(occlude_mask/255).unsqueeze(0).float()

            R = np.array(query['R']).astype(np.float32)
            T = np.array(query['T']).astype(np.float32)
            init_camera = GS_Camera2D(R=R.T, T=T, FoVx=FovX, FoVy=FovY,
                    image=image_tensor, colmap_id=0, uid=0, image_name=subbrand, gt_alpha_mask=None, data_device=device)

            query['occlude'] = occlude
            GS_Refiner(image_tensor, mask_tensor, omask_tensor, init_camera, gaussians=gaussians, device=device, bbox=bbox, query=query)
 
            pose = query['pose']
            loss = query['loss']
            track_id = query['track_id']
            name = query['name']
            if track_id not in dic:
                dic[track_id] = {}
            save_path = json_path[:-5] + '%s.png' % name
            rend_path = save_path.replace("poses_v3", "render")
            rgba_path = save_path.replace("poses_v3", "rgba")
            os.makedirs(os.path.dirname(rend_path), exist_ok=True)            
            os.makedirs(os.path.dirname(rgba_path), exist_ok=True)
            if name not in dic[track_id] or (name in dic[track_id] and loss[0] < dic[track_id][name]['loss'][0]):
                dic[track_id][name] = {'pose': pose, 'loss': loss}
                cv2.imwrite(save_path, query['cat'][...,::-1])
                cv2.imwrite(rend_path, query['render'][...,::-1])
                cv2.imwrite(rgba_path, rgba[...,[2,1,0,3]])
    
    with open(json_path, 'w') as f:
        json.dump(dic, f, indent=4, ensure_ascii=False)
    

def GS_Refiner(target_img, mask, omask, init_camera, gaussians, device=None, bbox=None, query=None):
    color = target_img[mask.expand_as(target_img) > 0.5].mean()
    black_bg = color > 0.5 or query['occlude']

    if black_bg:
        gaussian_BG = torch.zeros((3), device=device)
    else:
        gaussian_BG = torch.ones((3), device=device)
        target_img[mask.expand_as(target_img) < 0.5] = 1.0
    print("bg color: ", gaussian_BG)
    CFG.START_LR = 0.03
    gaussians.initialize_pose([init_camera]) # initialize 0
    x, y, w, h = bbox
    print(init_camera.image_name, [h, w], CFG.START_LR, query['occlude'], omask.max())
    optimizer = optim.AdamW([{'params': [gaussians._delta_R], 'lr': CFG.START_LR*0.5, 'name': 'rotation'}, 
                             {'params': [gaussians._delta_T], 'lr': CFG.START_LR, 'name': 'translation'}, ], lr=0.0)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, CFG.MAX_STEPS,
                                                warmup_steps=CFG.WARMUP, 
                                                max_lr=CFG.START_LR, 
                                                min_lr=CFG.END_LR)
    iter_losses = list()
    best_RT, min_loss = None, [1e6, 1e6]
    best_img = None
    # CFG.MAX_STEPS = 1000
    for iter_step in range(1, CFG.MAX_STEPS+1):  # 100
        if iter_step == CFG.MAX_STEPS * 1 / 2:
            for param_group in optimizer.param_groups:
                if param_group["name"] in ["translation", "rotation"]:
                    param_group['lr'] *= 0.1
                    
        idx = 0
        # ret = GS_Renderer(init_camera, gaussians, gaussian_PipeP, gaussian_BG)
        ret = GS_Renderer2D(init_camera, gaussians, gaussian_PipeP, gaussian_BG, idx=idx)
        unmasked = ret['render'].clone()
        render_img = ret['render']
        render_mask = ret['rend_alpha']
        if omask.max() > 0:
            render_img[omask.expand_as(render_img) >= 0.5] = 0 if black_bg else 1.0
            render_mask[omask.expand_as(render_mask) >= 0.5] = 0
        elif (iter_step >= CFG.MAX_STEPS * 0.4 and query['occlude']) or iter_step >= CFG.MAX_STEPS - 20:
            render_img[mask.expand_as(render_img) <= 0.5] = 0
            render_mask[mask.expand_as(render_mask) <= 0.5] = 0

        loss = 0
        if iter_step < CFG.MAX_STEPS * 0.4 and max(h, w) > 200:
            resz_h, resz_w = 600, 800
            def find_non1_region(mask):
                non1_indices = torch.where(mask > 0)
                min_y, min_x = torch.min(non1_indices[-2]), torch.min(non1_indices[-1])
                max_y, max_x = torch.max(non1_indices[-2]), torch.max(non1_indices[-1])
                return min_y.item(), min_x.item(), max_y.item(), max_x.item()
            with torch.inference_mode():
                bounds = find_non1_region(ret['rend_alpha'] + mask)
                croped_render_img = ret['render'][:, bounds[0]:bounds[2], bounds[1]:bounds[3]]
                resized_render_img = transforms.Resize((resz_h, resz_w))(croped_render_img)
                croped_target_img = target_img[:, bounds[0]:bounds[2], bounds[1]:bounds[3]]
                resized_target_img = transforms.Resize((resz_h, resz_w))(croped_target_img)
                input_dict = {
                    "image0": K.color.rgb_to_grayscale(resized_render_img[None, ...]),
                    "image1": K.color.rgb_to_grayscale(resized_target_img[None, ...])
                }
                correspondences = matcher(input_dict)
            mkpts0 = correspondences["keypoints0"].cpu().numpy()    # nx2, xy order
            mkpts1 = correspondences["keypoints1"].cpu().numpy()
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            inliers = inliers[:,0] > 0
            """# save for debug
            fig, ax = plt.subplots()

            draw_LAF_matches(
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts0).view(1, -1, 2),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1),
                ),
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts1).view(1, -1, 2),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                ),
                torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
                K.tensor_to_image(resized_render_img),
                K.tensor_to_image(resized_target_img),
                inliers,
                draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
                fig=fig,
                ax=ax,
            )
            fig.savefig('ansj/%s_%04d_matches.jpg' % (init_camera.image_name, iter_step))
            """
            ## mapping kpts to origin image
            crop_h, crop_w = croped_render_img.shape[-2:]
            mkpts0[:, 0] *= (crop_w / resz_w)
            mkpts0[:, 1] *= (crop_h / resz_h)
            mkpts1[:, 0] *= (crop_w / resz_w)
            mkpts1[:, 1] *= (crop_h / resz_h)
            mkpts0[:, 0] += bounds[1]
            mkpts0[:, 1] += bounds[0]
            mkpts1[:, 0] += bounds[1]
            mkpts1[:, 1] += bounds[0]
            mkpts0 = mkpts0[inliers,:]
            mkpts1 = mkpts1[inliers,:]
            mkpts0 = np.round(mkpts0).astype('int32')
            height, width = render_img.shape[-2:]
            uvs = ret['uvs'][:2]
            sample_uvs = uvs[:, mkpts0[:, 1], mkpts0[:, 0]]

            pred_uvs = torch.from_numpy(mkpts0).to(device).float().T
            pred_uvs[0] = pred_uvs[0] * 2 / width - 1.0
            pred_uvs[1] = pred_uvs[1] * 2 / height - 1.0

            gt_uvs = torch.from_numpy(mkpts1).to(device).float().T
            gt_uvs[0] = gt_uvs[0] * 2 / width - 1.0
            gt_uvs[1] = gt_uvs[1] * 2 / height - 1.0
            # set_trace()
            uv_mask = (sample_uvs.sum(dim=0) > 0)
            if uv_mask.sum() > 0:
                flow_loss = (torch.abs(sample_uvs - gt_uvs).sum(dim=0)[uv_mask]).mean()
                loss += flow_loss
            else:
                print("no inlier")

        union_mask = (mask + render_mask.detach()) > 0.5
        inter_mask = (mask * render_mask.detach()) > 0.5
        rgb_loss = MSELoss(render_img, target_img).sum()  / (union_mask.sum() + 1e-6)
        mask_loss = MSELoss(render_mask.float(), mask.float()).sum() / (union_mask.sum() + 1e-6)
        loss += (rgb_loss + mask_loss * 1.0)
        if CFG.USE_SSIM:
            loss  += (1 - SSIM_METRIC(render_img[None, ...], target_img[None, ...])) * 1.0
        
        masked_render = render_img.detach().clone()
        masked_render[union_mask.expand_as(render_img) <= 0.5] = 0 if black_bg else 1.0
        metric_loss = ((masked_render - target_img)**2).sum() / (union_mask.sum() + 1e-6)
        if metric_loss < min_loss[0] or (iter_step >= CFG.EARLY_STOP_MIN_STEPS and loss < min_loss[1]):
            min_loss = [metric_loss.item(), loss.item()]
            best_img = torch.cat([ret['rend_alpha'].detach(), unmasked], dim=0)
            best_cat = (unmasked.detach()+target_img.detach())*0.5
            best_RT = gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()
        
        if iter_step % 10 == 0:
            cat = ((unmasked.detach()+target_img.detach())*0.5).permute(1,2,0).cpu().numpy() * 255
            # save_path = 'ansj/%s_%04d.jpg' % (init_camera.image_name, iter_step/10)
            save_path = 'ansj/%s_%04d.jpg' % (init_camera.image_name, iter_step)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cat[...,::-1])
        
        total_loss = loss.item()
        total_metric = metric_loss.item()
        
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
        
        iter_losses.append(total_loss)
        if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
            loss_grads = (torch.as_tensor(iter_losses)[1:] - torch.as_tensor(iter_losses)[:-1]).abs()
            if loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].max() < CFG.EARLY_STOP_LOSS_GRAD_NORM: # early stop the refinement
                break
        
        if iter_step % 10 == 0:
            print(iter_step, total_loss, total_metric)
        
        torch.cuda.empty_cache()
    
    print(iter_step, total_loss, total_metric)
    
    query['pose'] = best_RT.tolist()
    query['loss'] = min_loss
    query['cat'] = best_cat.permute(1,2,0).cpu().numpy() * 255
    query['render'] = best_img.detach().permute(1,2,0).cpu().numpy() * 255

    return query

def find_contour(alpha, bbox):
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    max_len = 0
    contour = None
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if e < s: e += len(max_contour)
        
        length = (e - s)
        if length > max_len:
            max_len = length
            contour = max_contour[s:e+1]
            start = max_contour[s][0]
            end = max_contour[e][0]
            far = max_contour[f][0]
    
    x, y, w, h = bbox
    k1 = (start[1] - far[1]) / (start[0] - far[0])
    
    return contour


def isinside(alpha):
    height, width = alpha.shape
    x, y, w, h = cv2.boundingRect(alpha)
    if ((x <= 0 or x+w >= width-1) and alpha[:,x].sum() > 10) or \
       ((y <= 0 or y+h >= height-1) and alpha[y,:].sum() > 10):
           return False
    return True


def classify_quadrilateral(vertices):
    """
    根据四边形的四个顶点，判断左上、右上、右下、左下的位置。
    :param vertices: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 四个顶点坐标
    :return: {'top_left': (x, y), 'top_right': (x, y), 'bottom_right': (x, y), 'bottom_left': (x, y)}
    """
    # 按 y 值从小到大排序，如果 y 值相同，则按 x 值排序
    vertices = sorted(vertices, key=lambda p: (p[1], p[0]))
    
    # 分成上两个点和下两个点
    top_points = vertices[:2]
    bottom_points = vertices[2:]
    
    # 上两个点中 x 较小的是左上，较大的是右上
    top_left, top_right = sorted(top_points, key=lambda p: p[0])
    # 下两个点中 x 较小的是左下，较大的是右下
    bottom_left, bottom_right = sorted(bottom_points, key=lambda p: p[0])
    
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_right': bottom_right,
        'bottom_left': bottom_left
    }

def adjust_coordinates(vertices, threshold=10):
    """
    根据条件调整左上和右上的横坐标。
    :param vertices: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 四个顶点坐标
    :return: 调整后的坐标字典
    """
    # 分类顶点
    points = classify_quadrilateral(vertices)
    top_left = points['top_left']
    top_right = points['top_right']
    bottom_right = points['bottom_right']
    bottom_left = points['bottom_left']
    
    # 计算水平距离
    top_distance = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    bottom_distance = np.sqrt((bottom_right[0] - bottom_left[0])**2 + (bottom_right[1] - bottom_left[1])**2)

    if top_distance < bottom_distance:
        threshold = min(top_distance, threshold)
    else:
        threshold = min(top_distance, max(top_distance - bottom_distance, threshold))
    
    # 条件判断并调整坐标
    if bottom_distance - top_distance < threshold:
        adjustment = threshold/2
        # 调整左上和右上的横坐标
        # top_left = [top_left[0] + adjustment, top_left[1]]
        # top_right = [top_right[0] - adjustment, top_right[1]]
        bottom_left = [bottom_left[0] - adjustment, bottom_left[1]]
        bottom_right = [bottom_right[0] + adjustment, bottom_right[1]]
    
    # 返回调整后的坐标
    return [top_left, top_right, bottom_right, bottom_left]

def detectAndMatch(image1, image2, ratio=0.75):
    # detect and extract features from the image
    # descriptor = cv2.xfeatures2d.SURF_create()
    # descriptor = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.SIFT_create()
    (kps1, featuresA) = descriptor.detectAndCompute(image1, None)
    (kps2, featuresB) = descriptor.detectAndCompute(image2, None)

    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    
    matches = []
    mkpts0 = []
    mkpts1 = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx, m))
            mkpts0.append(kps1[m[0].queryIdx])
            mkpts1.append(kps2[m[0].trainIdx])
            
    # return a tuple of keypoints and features
    return np.array(mkpts0), np.array(mkpts1)

def search_car_brand(sfm_list, brand, subbrand, color):
    sfm_paths = []
    for sfm_path in sfm_list:
        if brand in sfm_path and subbrand in sfm_path and color in sfm_path:
            img_paths = os.listdir(os.path.join(sfm_path.replace('2dgs', 'sfm'), 'images'))
            if len(img_paths) < 40: continue
            sfm_paths.append(sfm_path)
    
    return sfm_paths

def brand_list(path):
    sfm_paths = []
    for dir in os.listdir(path):
        car_dir = os.path.join(path, dir)
        for color in os.listdir(car_dir):
            sparse_dir = os.path.join(car_dir, color)
            sfm_paths.append(sparse_dir)
    
    return sfm_paths
    

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


if __name__ == "__main__":
    main()
