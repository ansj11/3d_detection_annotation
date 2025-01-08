import os
import cv2
import json
import argparse
import threading
import subprocess
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Lock
from pdb import set_trace
import matplotlib.pyplot as plt

from gaussian_2dgs.gaussian_model import GaussianModel as GaussianModel2D
from gaussian_object.utils.graphics_utils import focal2fov



def xywh2xyxy(bbox, height, width):
    x0, y0, w, h = bbox
    x1 = min(x0 + w, width)
    y1 = min(y0 + h, height)
    
    return [int(x) for x in [x0, y0, x1, y1]]


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

def get_kpts3d(kpts3d_dict, brand, subbrand):
    for key in kpts3d_dict.keys():
        if brand in key and subbrand in key:
            return np.array(kpts3d_dict[key])
    return None

"""process all objects without occlusion"""
def main(debug=True):
    root = "/gemini/data-2/segment/"

    # path = os.path.join(root, 'infos/infos_dict_v2.json')
    path = os.path.join(root, 'infos_v1/infos_dict_v1.json')
    with open(path, 'r') as f:
        infos_dict = json.load(f)

    path = os.path.join(root, 'infos/kpts3d_dict.json')
    with open(path, 'r') as f:
        kpts3d_dict = json.load(f)

    # occlude_path = "metaloop_20241126205435/occluded.json"
    occlude_path = "metaloop_20241126210108/occluded.json"
    with open(occlude_path, 'r') as f:
        occlude_dict = json.load(f)
    
    fx, fy = 3630, 3600
    cx, cy = 1344.0, 760.0
    cameraMatrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0, 1.]
            ], dtype=np.float32)
    
    totoal_num = len(infos_dict)
    count, success = 0, 0
    mask2idx = {}
    for img_path in infos_dict.keys():
        key = img_path.replace('./segment', '')
        basename = os.path.basename(img_path).split('.')[0]
        print("process: (%d %d) / %d, %s......" % (count, success, totoal_num, img_path))
        count += 1
        occlude_list = occlude_dict[basename]

        track_dict = infos_dict[key]
    
        image = cv2.imread(img_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # mask_path = img_path.replace('trial_v2', 'masks').replace('jpg', 'png')
        mask_path = img_path.replace('trial_v1', 'masks').replace('jpg', 'png')
        mask = cv2.imread(mask_path, 0)
    
        sfm_path = os.path.join(root, 'cars/2dgs')
        sfm_list = brand_list(sfm_path)
        FovX = focal2fov(cameraMatrix[0,0], width)
        FovY = focal2fov(cameraMatrix[1,1], height)
    
        query_dict = {}
        render_image = image.copy()
        for track_id, result in track_dict.items():
            color = result['color']
            bbox = result['bbox']
            brand, subbrand, year = result['brand']
            kpts2d = result['kpts']
            kpts2d = np.array(kpts2d)
            kpts2d[:,0] += bbox[0]
            kpts2d[:,1] += bbox[1]
            brand_str = '%s_%s_%s' % (brand, subbrand, year)
            if max(bbox[-2:]) <= 50:
                continue

            print(bbox[-2:], brand_str)
            xyxy = xywh2xyxy(bbox, height, width)
            x0, y0, x1, y1 = xyxy
            alpha, max_idx = get_mask2(mask, xyxy)
            
            if not isinside(alpha) or alpha.max() == 0:
                print('not inside or no mask', brand_str, alpha.max())
                continue

            occlude = False
            dilate = cv2.dilate(alpha[y0:y1, x0:x1], np.ones((3,3), np.uint8), iterations=2)
            indices = np.unique(mask[y0:y1, x0:x1][dilate > 0])
            
            if len(indices) > 2 or max_idx in occlude_list:
                print("is occluded mask", brand_str)
                occlude = True
            if max_idx in occlude_list:
                print("in occluded list", brand_str)
                occlude = True
            
            if basename not in mask2idx:
                mask2idx[basename] = {}
            mask2idx[basename][track_id] = int(max_idx)
            
            # pc_dirs = search_car_brand(sfm_list, brand, subbrand, color)
            # if (len(pc_dirs) == 0):
            #     print("not exists 2dgs...", brand_str)
            #     continue
            # queries = []
            # for pc_dir in pc_dirs:
            #     if not os.path.exists(pc_dir):
            #         print(pc_dir, 'not exists...', brand_str)
            #         continue

            #     pc_path = os.path.join(pc_dir, 'point_cloud/iteration_7000/point_cloud.ply')
            #     gs_dir = '_'.join(pc_dir.split('/')[-2:])
            #     if not os.path.exists(pc_path):
            #         print("not exists...", pc_path, brand_str)
            #         continue

            #     brand_key = pc_dir.split('/')[-2]
            #     if brand_key not in kpts3d_dict:
            #         print("not exists kpts3d...", key)
            #         continue
            #     kpts3d = np.array(kpts3d_dict[brand_key])
                
            #     distCoeffs = np.zeros((5), dtype=np.float32)
            #     for i, pt in enumerate(kpts2d):
            #         x, y = np.round(pt).astype('int')
            #         cv2.circle(render_image, (x, y), 3, (255, 0, 0), -1)
            #     flag, rvec, tvec = cv2.solvePnP(kpts3d, kpts2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            #     flag, rvecs, tvecs, ret = cv2.solvePnPGeneric(kpts3d[[3,0,1,2]], kpts2d[[3,0,1,2]], cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            #     rvecs = list(rvecs) + [rvec]
            #     tvecs = list(tvecs) + [tvec]
            #     best_R, best_T = None, None
            #     best_iou, best_mask, best_dist = 0.0, None, 1e6
            #     ys, xs = np.where(alpha > 0.5)
            #     centerx, centery = xs.mean(), ys.mean()
            #     gaussians = GaussianModel2D(3)
            #     gaussians.load_ply([pc_path])
            #     for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            #         R, _ = cv2.Rodrigues(rvec)
            #         T = tvec / 1000.0
            #         render_mask = np.zeros_like(mask)
            #         xyz = gaussians._xyz[0].detach().cpu().numpy() @ R.T + T[:,0]
            #         ix = xyz[:,0] / xyz[:,-1] * cameraMatrix[0,0] + cx
            #         iy = xyz[:,1] / xyz[:,-1] * cameraMatrix[1,1] + cy
            #         valid_uv = (ix >= 0) * (ix < width) * (iy >= 0) * (iy < height)
            #         if valid_uv.sum() == 0:
            #             print("project point not in image")
            #             continue
            #         ix, iy = ix[valid_uv].astype('int'), iy[valid_uv].astype('int')
            #         try:
            #             render_mask[iy, ix] = 1.0
            #         except: set_trace()
            #         overlap = (mask * render_mask).sum()
            #         iou = overlap / (mask.sum() + render_mask.sum() - overlap + 1e-6)
            #         renderx, rendery = ix.mean(), iy.mean()
            #         dist = np.sqrt((centerx - renderx)**2 + (centery - rendery)**2)
            #         if iou > best_iou or (dist < best_dist and iou < 0.1):
            #             best_iou = iou
            #             best_dist = dist
            #             best_R, best_T = R, T
            #             best_mask = render_mask
            #     if best_T is None: 
            #         print("no valid pose", brand_str)
            #         continue
                
            #     query = {'R': best_R.tolist(), 'T': best_T[:,0].tolist(), 'gs_path': pc_dir,
            #             'track_id': track_id, 'kpts3d': kpts3d.tolist(), 'name': gs_dir}
            #     queries.append(query)
            # if (len(queries) == 0):
            #     print("no valid query", brand_str)
            #     continue
            # render_image[best_mask > 0.5,:] = render_image[best_mask > 0.5,:] * 0.5 + 0.5 * np.array([0, 255, 0])
            # query_dict[track_id] = {
            #     'fovxy': [FovX, FovY],
            #     'bbox': bbox,
            #     'xyxy': xyxy,
            #     'brand': [brand, subbrand, year, color],
            #     'kpts2d': kpts2d.tolist(),
            #     'occlude': occlude,
            #     'queries': queries,}

        # if len(query_dict) == 0:
        #     print("no valid query", img_path)
        #     continue
        success += 1
        
    json_path = occlude_path.replace('occluded.json', 'mask2idx.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(mask2idx, f, indent=4, ensure_ascii=False)
    print(success, totoal_num)

def isinside(alpha):
    height, width = alpha.shape
    x, y, w, h = cv2.boundingRect(alpha)
    if (x <= 0 and alpha[:,x].sum() >= w) or (x+w >= width-1 and alpha[:,x+w-1].sum() >= w) or \
       (y <= 0 and alpha[y,:].sum() >= h) or (y+h >= height-1 and alpha[y+h-1,:].sum() >= h):
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
            sfm_paths.append([sfm_path, len(img_paths)])
    
    if len(sfm_paths) == 0:
        return sfm_paths
    sfm_paths = sorted(sfm_paths, key=lambda x: x[1], reverse=True)
    max_len = sfm_paths[0][1]
    paths = [sfm_path[0] for sfm_path in sfm_paths if sfm_path[1] == max_len]
    
    return paths


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
