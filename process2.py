import os
import cv2
import json
import argparse
import threading
import subprocess
import numpy as np
from tqdm import tqdm
from rembg import remove
from multiprocessing import Process, Lock
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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from gaussian_2dgs.arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision.transforms as transforms

# 2dgs setting 
from gaussian_2dgs.gaussian_model import GaussianModel as GaussianModel2D
from gaussian_2dgs.gaussian_renderer import render as GS_Renderer2D
from gaussian_2dgs.scene.cameras import Camera as GS_Camera2D
from gaussian_2dgs.utils.graphics_utils import focal2fov as focal2fov2D

from misc_utils.losses import ScaleAndShiftInvariantLoss
from transformers import pipeline
from PIL import Image
import PIL
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
parser.add_argument("--fx", type=str, default=3690,
                        help="input path")
parser.add_argument("--fy", type=str, default=3690,
                        help="input path")
args = parser.parse_args()

MSELoss = torch.nn.MSELoss(reduction='mean')
L1Loss = torch.nn.L1Loss(reduction='mean')
SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)
SSILoss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)


def get_bbox(bbox, scale, height, width):
    x0, y0, w, h = bbox
    maxhw = max(h, w) * scale
    th, tw = maxhw, maxhw
    cx, cy = x0+w/2, y0+h/2
    x0 = int(max(0, cx - tw/2))
    x1 = int(min(width, cx + tw/2))
    y0 = int(max(0, cy - th/2))
    y1 = int(min(height-1, cy + th/2))
    
    return x0, y0, x1, y1

def xywh2xyxy(bbox, height, width):
    x0, y0, w, h = bbox
    x1 = min(x0 + w, width)
    y1 = min(y0 + h, height)
    
    return [int(x) for x in [x0, y0, x1, y1]]

def get_mask(mask):
    indices = np.unique(mask)
    max_area = 0
    max_idx = -1
    for idx in indices:
        if idx == 0:
            continue
        area = (mask == idx).sum()
        if area > max_area:
            max_area = area
            max_idx = idx

    return (mask == max_idx).astype(np.uint8) * 255


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
        if iou > max_area:
            max_area = iou
            max_idx = idx

    return (mask == max_idx).astype(np.uint8) * 255

def get_kpts3d(kpts3d_dict, brand, subbrand):
    for key in kpts3d_dict.keys():
        if brand in key and subbrand in key:
            return np.array(kpts3d_dict[key])
    return None

"""process normal objects with occlusion"""
def main(mode='bbox', debug=True, padding_pixel=0):
    root = "/gemini/data-2/segment/"

    path = os.path.join(root, 'infos/infos_dict_v2.json')
    with open(path, 'r') as f:
        infos_dict = json.load(f)

    img_path = args.input
    key = img_path.replace('./segment', '')
    basename = os.path.basename(img_path).split('.')[0]
    if key not in infos_dict:
        print('image not in infos_dict', key)
        return

    path = os.path.join(root, 'infos/kpts3d_dict.json')
    with open(path, 'r') as f:
        kpts3d_dict = json.load(f)
        
    track_dict = infos_dict[key]
    
    image = cv2.imread(img_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    mask_path = img_path.replace('trial_v2', 'masks').replace('jpg', 'png')
    mask = cv2.imread(mask_path, 0)
    
    sfm_path = os.path.join(root, 'cars/2dgs')
    sfm_list = brand_list(sfm_path)
    fx, fy = args.fx, args.fy
    cx, cy = 1344.0, 760.0
    device = torch.device('cuda')
    
    cameras, queries = [], []
    images, masks, bboxes, gs_paths = [], [], [], []
    for track_id, result in track_dict.items():
        color = result['color']
        bbox = result['bbox']
        brand, subbrand, year = result['brand']
        # if brand == '奇瑞': set_trace()
        # kpts2d = adjust_coordinates(result['kpts'])
        kpts2d = result['kpts']
        kpts2d = np.array(kpts2d)
        kpts2d[:,0] += bbox[0]
        kpts2d[:,1] += bbox[1]
        brand_str = '%s_%s_%s' % (brand, subbrand, year)
        if max(bbox[-2:]) <= 160:
             continue
        # if brand not in ['领克', '哈弗', '宝马', '日产']:
        #     continue
        # if subbrand not in ['宝马3系', '轩逸']:
        #     continue
        print(bbox[-2:], brand_str)
        xyxy = xywh2xyxy(bbox, height, width)
        x0, y0, x1, y1 = xyxy
        crop = image[y0:y1, x0:x1]
        rgba = np.zeros((height, width, 4), dtype='uint8')
        rgba[y0:y1, x0:x1, :3] = crop
        alpha = get_mask2(mask, xyxy)
        rgba[..., -1] = alpha
        # if "一汽_森雅R7_2017款" != brand_str:
        #     continue
        # dilate = cv2.dilate(alpha[y0:y1, x0:x1], np.ones((3,3), np.uint8), iterations=2)
        # indices = np.unique(mask[y0:y1, x0:x1][dilate > 0])
        # if len(indices) <= 2:
        #     print("not occluded mask", brand_str)
        #     continue
        
        pc_dirs = search_car_brand(sfm_list, brand, subbrand, color)
        # if brand == '比亚迪':
        #     # pc_dirs = ["/gemini/data-2/cars/2dgs/比亚迪-秦PLUS_2021款-DM-i-55KM-旗舰型/雪域白"]
        #     pc_dirs = ["/gemini/data-2/cars/2dgs/比亚迪-秦PLUS_2023款-冠军版-EV-610KM卓越型/雪域白"]
        for pc_dir in pc_dirs:
            if not os.path.exists(pc_dir):
                print(pc_dir, 'not exists...', brand_str)
                continue

            pc_path = os.path.join(pc_dir, 'point_cloud/iteration_30000/point_cloud.ply')
            gs_dir = '_'.join(pc_dir.split('/')[-2:])
            if not os.path.exists(pc_path):
                print("not exists...", pc_path, brand_str)
                continue

            brand_key = pc_dir.split('/')[-2]
            if brand_key not in kpts3d_dict:
                print("not exists kpts3d...", key)
                continue
            kpts3d = np.array(kpts3d_dict[brand_key])
            
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
                kpts2d[:,0] += left
                kpts2d[:,1] += top
                xyxy = [xyxy[0] + left, xyxy[1] + top, xyxy[2] + left, xyxy[3] + top]
            
            rgba_tensor = torch.from_numpy(rgba/255.0).to(device).permute(2, 0, 1)
            mask_tensor = rgba_tensor[-1:].float()
            image_tensor = rgba_tensor[:-1].float() * mask_tensor
            
            query = {'xywh': bbox, 'xyxy': xyxy, 'image': image, 'brand': [brand, subbrand, color], 
                    'track_id': track_id, 'kpts2d': kpts2d, 'kpts3d': kpts3d, 'name': gs_dir}
            queries.append(query)
            images.append(image_tensor)
            masks.append(mask_tensor)
            bboxes.append(bbox)
            gs_paths.append(pc_path)

    # gaussians = GaussianModel()
    if len(queries) == 0:
        print("no valid query", img_path)
        return
    print(len(gs_paths), [query['name'] for query in queries])
    gaussians = GaussianModel2D(3)
    gaussians.load_ply(gs_paths)

    cameraMatrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0, 1.]
        ], dtype=np.float32)
    FovX = focal2fov(cameraMatrix[0,0], pW)
    FovY = focal2fov(cameraMatrix[1,1], pH)
    init_cameras = []
    image = (sum(images) / (sum(masks)+1e-6)).permute(1,2,0).cpu().numpy()
    render_image = image.copy() * 255
    for idx, query in enumerate(queries):
        distCoeffs = np.zeros((5), dtype=np.float32)
        points2d = query['kpts2d']
        points3d = query['kpts3d']
        subbrand = query['brand'][1]
        for i, pt in enumerate(points2d):
            x, y = np.round(pt).astype('int')
            cv2.circle(render_image, (x, y), 3, (255, 0, 0), -1)
        flag, rvecs, tvecs, ret = cv2.solvePnPGeneric(points3d[[3,0,1,2]], points2d[[3,0,1,2]], cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        best_R, best_T = None, None
        best_iou, best_mask, best_dist = 0.0, None, 1e6
        mask = masks[idx].squeeze().cpu().numpy()
        ys, xs = np.where(mask > 0.5)
        centerx, centery = xs.mean(), ys.mean()
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            T = tvec / 1000.0
            render_mask = np.zeros_like(mask)
            xyz = gaussians._xyz[idx].detach().cpu().numpy() @ R.T + T[:,0]
            ix = xyz[:,0] / xyz[:,-1] * cameraMatrix[0,0] + cx
            iy = xyz[:,1] / xyz[:,-1] * cameraMatrix[1,1] + cy
            valid_uv = (ix >= 0) * (ix < pW) * (iy >= 0) * (iy < pH)
            if valid_uv.sum() == 0:
                print("project point not in image")
                continue
            ix, iy = ix[valid_uv].astype('int'), iy[valid_uv].astype('int')
            try:
                render_mask[iy, ix] = 1.0
            except: set_trace()
            overlap = (mask * render_mask).sum()
            iou = overlap / (mask.sum() + render_mask.sum() - overlap + 1e-6)
            renderx, rendery = ix.mean(), iy.mean()
            dist = np.sqrt((centerx - renderx)**2 + (centery - rendery)**2)
            if iou > best_iou or (dist < best_dist and iou < 0.1):
                best_iou = iou
                best_dist = dist
                best_R, best_T = R, T
                best_mask = render_mask
        if best_T is None: print("no valid pose", query['brand'])
        render_image[best_mask > 0.5,:] = render_image[best_mask > 0.5,:] * 0.5 + 0.5 * np.array([0, 255, 0])
        R, T = best_R, best_T
        init_camera = GS_Camera2D(R=R.T, T=T[:,0], FoVx=FovX, FoVy=FovY,
                    image=image_tensor, colmap_id=0, uid=0, image_name=subbrand, gt_alpha_mask=None, data_device=device)
        init_cameras.append(init_camera)
    if debug:
        save_path = 'ansj/%s_%d_%d_%04d.png' % (basename, fx, fy, 0)
        cv2.imwrite(save_path, render_image[...,::-1])
    ret = GS_Refiner(images, masks, init_cameras, gaussians=gaussians, device=device, bboxes=bboxes, scale=[fx, fy])

    json_path = img_path.replace('trial_v2', 'poses_v2')[:-3] + 'json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    dic = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            dic = json.load(f)
    for idx, query in enumerate(queries):
        pose = ret['RT'][idx].tolist()
        track_id = query['track_id']
        name = query['name']
        if track_id not in dic:
            dic[track_id] = {}
        if name not in dic[track_id] or dic[track_id][name]['loss'] > ret['loss'][idx]:
            dic[track_id][name] = {'pose': pose, 'loss': ret['loss'][idx]}
            cv2.imwrite(json_path[:-5] + '%s.png' % name, ret['cat_img'][idx][...,::-1])
    
    with open(json_path, 'w') as f:
        json.dump(dic, f, indent=4, ensure_ascii=False)
    

def GS_Refiner(images, masks, init_cameras, gaussians, device=None, bboxes=None, mode='gspose', scale=1.0):
    
    gaussian_BG = torch.zeros((3), device=device)
    
    gaussians.initialize_pose(init_cameras) # initialize 0
    # distort_model = ImageDistortionModel(device=device)
    print(scale, init_cameras[0].world_view_transform[-1,:3])
    optimizer = optim.AdamW([{'params': [gaussians._delta_R], 'lr': CFG.START_LR*0.5, 'name': 'rotation'}, 
                             {'params': [gaussians._delta_T], 'lr': CFG.START_LR, 'name': 'translation'}, ], lr=0.0)
    # lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, CFG.MAX_STEPS,
    #                                             warmup_steps=CFG.WARMUP, 
    #                                             max_lr=CFG.START_LR, 
    #                                             min_lr=CFG.END_LR)
    iter_losses = list()
    MAX_STEPS = max(1000, CFG.MAX_STEPS)
    best_RTs, min_losses = [None for _ in range(len(images))], [1e6 for _ in range(len(images))]
    best_imgs = [None for _ in range(len(images))]
    for iter_step in range(MAX_STEPS):  # 100
        if iter_step == MAX_STEPS * 1 / 2:
            for param_group in optimizer.param_groups:
                if param_group["name"] == "rotation":
                    param_group['lr'] = CFG.START_LR*0.1
                if param_group["name"] == "intrinsic":
                    param_group['lr'] = CFG.START_LR*0.0

        total_loss, total_metric = 0, 0
        update_cameras, losses = [], []
        for idx, (target_img, mask, init_camera) in enumerate(zip(images, masks, init_cameras)):
            # ret = GS_Renderer(init_camera, gaussians, gaussian_PipeP, gaussian_BG)
            ret = GS_Renderer2D(init_camera, gaussians, gaussian_PipeP, gaussian_BG, idx=idx)
            unmasked = ret['render'].clone()
            if iter_step >= MAX_STEPS * 0.2:
                ret['render'][masks[idx].expand_as(ret['render']) <= 0.5] = 0
                ret['rend_alpha'][masks[idx].expand_as(ret['rend_alpha']) <= 0.5] = 0
            render_img = ret['render']
            render_depth = ret['surf_depth']
            render_mask = ret['rend_alpha']

            if CFG.USE_MATCH:
                resz_h, resz_w = 600, 800
                with torch.inference_mode():
                    def find_non1_region(mask):
                        non1_indices = torch.where(mask > 0)
                        min_y, min_x = torch.min(non1_indices[-2]), torch.min(non1_indices[-1])
                        max_y, max_x = torch.max(non1_indices[-2]), torch.max(non1_indices[-1])
                        return min_y.item(), min_x.item(), max_y.item(), max_x.item()
                    bounds = find_non1_region(render_mask + mask)
                    croped_render_img = render_img[:, bounds[0]:bounds[2], bounds[1]:bounds[3]]
                    resized_render_img = transforms.Resize((resz_w, resz_h))(croped_render_img)
                    croped_target_img = target_img[:, bounds[0]:bounds[2], bounds[1]:bounds[3]]
                    resized_target_img = transforms.Resize((resz_w, resz_h))(croped_target_img)
                    input_dict = {
                        "image0": K.color.rgb_to_grayscale(resized_render_img[None, ...]),
                        "image1": K.color.rgb_to_grayscale(resized_target_img[None, ...])
                    }
                    correspondences = matcher(input_dict)
                mkpts0 = correspondences["keypoints0"].cpu().numpy()    # nx2, xy order
                mkpts1 = correspondences["keypoints1"].cpu().numpy()
                Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
                inliers = inliers > 0
                
                # save for debug
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
                
                ## mapping kpts to 3d points
                height, width = target_img.shape[-2:]
                mkpts0_3d = np.zeros((len(mkpts0), 3), dtype=np.float32)
                render_depth_np = render_depth.detach().cpu().numpy()[0]
                fx = width / 2 / gaussians._intrinsic[0].detach().cpu().numpy() # np.tan(init_camera.FoVx / 2)
                fy = height / 2 / gaussians._intrinsic[1].detach().cpu().numpy() # np.tan(init_camera.FoVy / 2)
                cx, cy = width / 2, height / 2
                for i, (u, v) in enumerate(mkpts0):
                    try:
                        z = render_depth_np[int(v), int(u)]
                    except: set_trace()
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    mkpts0_3d[i] = [x, y, z]
                
                ## calculate pnp pose
                cameraMatrix = np.array([
                                [fx, 0, cx],
                                [0, fy, cy],
                                [0,  0, 1.]
                            ], dtype=np.float32)
                distCoeffs = np.zeros((5), dtype=np.float32)
                flag, rvec, tvec = cv2.solvePnP(mkpts0_3d, mkpts1, cameraMatrix, distCoeffs)   # failed         
                flag, rvec, tvec, inliers = cv2.solvePnPRansac(mkpts0_3d, mkpts1, cameraMatrix, distCoeffs,
                                                        iterationsCount=100, reprojectionError=1.0, confidence=0.999,
                                                        #  iterationsCount=100, reprojectionError=0.8, confidence=0.8,
                                                        flags=cv2.SOLVEPNP_ITERATIVE)
                if not flag:
                    print("PnP pose estimation failed")
                    break
                R, _ = cv2.Rodrigues(rvec)
                RT = np.matmul(init_camera.R, R.T)  # (x * R + T) * dR' + dT, R=cam.R
                T = np.matmul(init_camera.T, R.T) + tvec[:,0]
                print(iter_step, rvec, tvec)
                init_camera = GS_Camera2D(T=T, R=RT,
                                    FoVx=init_camera.FoVx, FoVy=init_camera.FoVy,# cx_offset=0, cy_offset=0,
                                    image=target_img, colmap_id=0, uid=0, image_name=init_camera.image_name, gt_alpha_mask=None, data_device=device)
                update_cameras.append(init_camera)
                continue
        
            rgb_loss = MSELoss(render_img, target_img).mean() * 1
            mask_loss = MSELoss(render_mask.float(), mask.float()).mean() * 0.5
            # depth_loss = SSILoss(render_depth, target_depth, torch.ones_like(target_depth)).mean()*0.0
            loss = (rgb_loss + mask_loss) # + depth_loss
            if CFG.USE_SSIM:
                loss  += (1 - SSIM_METRIC(render_img[None, ...], target_img[None, ...])) * 1.0
            if CFG.USE_MS_SSIM:
                loss += (1 - MS_SSIM_METRIC(render_img[None, ...], target_img[None, ...])) * 1.0
            
            masked_render = render_img.detach().clone()
            masked_render[mask.expand_as(render_img) <= 0.5] = 0
            metric_loss = MSELoss(masked_render, target_img).mean() + (1 - SSIM_METRIC(masked_render[None, ...], target_img[None, ...]))
            cat = ((unmasked.detach()+target_img.detach())*0.5).permute(1,2,0).cpu().numpy() * 255
            if metric_loss < min_losses[idx]:
                min_losses[idx] = metric_loss.item()
                best_imgs[idx] = cat
                best_RTs[idx] = gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()[idx]
            
            if iter_step % 10 == 0:
                # save_path = 'ansj/%s_%04d.jpg' % (init_camera.image_name, iter_step/10)
                save_path = 'ansj/%s_%04d_%d.jpg' % (init_camera.image_name, iter_step, idx)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cat[...,::-1])
            
            losses.append(loss.item())
            total_loss += loss.item()
            total_metric += metric_loss.item()
            
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            
        if CFG.USE_MATCH:
            gaussians.initialize_pose(update_cameras) # update pose
            continue

        iter_losses.append(total_loss)
        if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
            loss_grads = (torch.as_tensor(iter_losses)[1:] - torch.as_tensor(iter_losses)[:-1]).abs()
            if loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].max() < CFG.EARLY_STOP_LOSS_GRAD_NORM: # early stop the refinement
                break
        
        if iter_step % 10 == 0:
            print(iter_step, total_loss, total_metric)
    
    height, width = target_img.shape[-2:]
    print(iter_step, total_loss, total_metric, gaussians._intrinsic.detach()[0].item()*width/2, gaussians._intrinsic.detach()[1].item()*height/2)
    
    # conduct output debug images as video
    frame_prefix = init_camera.image_name
    image_list = [f'ansj/{frame_prefix}_{i:04d}.jpg' for i in range(iter_step) if os.path.exists(f'ansj/{frame_prefix}_{i:04d}.jpg')]
    video_path = f'ansj/{frame_prefix}.mp4'
    # make_video:
    # if len(image_list) > 0:
    #     # cmd = ['ffmpeg', '-y', '-r', '10', '-i', f'ansj/{frame_prefix}_%04d_matches.jpg', '-c:v', 'libx264', '-vf', 'fps=25', video_path]
    #     cmd = ['ffmpeg', '-y', '-r', '10', '-i', f'ansj/{frame_prefix}_%04d.jpg', '-vf', 'fps=25', video_path]
    #     subprocess.run(cmd)

    outp = {
        'RT': np.stack(best_RTs),
        'iter_step': iter_step,
        'cat_img': best_imgs,
        'loss': min_losses
    }
    # print(iter_step, loss.item(), gaussians._delta_R, gaussians._delta_T, gaussians._intrinsic)

    return outp


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
