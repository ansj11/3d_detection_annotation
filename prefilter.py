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

    path = os.path.join(root, 'infos/infos_dict_v2.json')
    with open(path, 'r') as f:
        infos_dict = json.load(f)

    path = os.path.join(root, 'infos/kpts3d_dict.json')
    with open(path, 'r') as f:
        kpts3d_dict = json.load(f)

    occlude_path = "metaloop_20241126205435/occluded.json"
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
    count, success, total, small = 0, 0, 0, 0
    no2dgs, no3dkpts, noquery, nodict = 0, 0, 0, 0
    nomask, nopose, valid, nobrand = 0, 0, 0, 0
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
        mask_path = img_path.replace('trial_v2', 'masks').replace('jpg', 'png')
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
            total += 1
            if max(bbox[-2:]) <= 50:
                small += 1
                continue
            if brand == 'null':
                print("no brand: ", brand_str)
                nobrand += 1
                continue

            print(bbox[-2:], brand_str)
            xyxy = xywh2xyxy(bbox, height, width)
            x0, y0, x1, y1 = xyxy
            alpha, max_idx = get_mask2(mask, xyxy)
            
            if not isinside(alpha) or alpha.max() == 0:
                print('not inside or no mask', brand_str, alpha.max())
                nomask += 1
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
            
            pc_dirs = search_car_brand(sfm_list, brand, subbrand, color)
            if (len(pc_dirs) == 0):
                print("not exists 2dgs...", brand_str)
                no2dgs += 1
                continue
            gs_dirs = []
            for pc_dir in pc_dirs:
                pc_path = os.path.join(pc_dir, 'point_cloud/iteration_7000/point_cloud.ply')
                gs_dir = '_'.join(pc_dir.split('/')[-2:])
                if not os.path.exists(pc_path):
                    print("not exists...", pc_path, brand_str)
                    continue
                gs_dirs.append(pc_dir)
            pc_dirs = gs_dirs
            if (len(pc_dirs) == 0):
                print("not exists 2dgs...", brand_str)
                no2dgs += 1
                continue
            
            queries = []
            for pc_dir in pc_dirs:
                if not os.path.exists(pc_dir):
                    print(pc_dir, 'not exists...', brand_str)
                    continue

                pc_path = os.path.join(pc_dir, 'point_cloud/iteration_7000/point_cloud.ply')
                gs_dir = '_'.join(pc_dir.split('/')[-2:])
                if not os.path.exists(pc_path):
                    print("not exists...", pc_path, brand_str)
                    continue

                brand_key = pc_dir.split('/')[-2]
                if brand_key not in kpts3d_dict:
                    print("not exists kpts3d...", key)
                    no3dkpts += 1
                    continue
                kpts3d = np.array(kpts3d_dict[brand_key])
                
                distCoeffs = np.zeros((5), dtype=np.float32)
                for i, pt in enumerate(kpts2d):
                    x, y = np.round(pt).astype('int')
                    cv2.circle(render_image, (x, y), 3, (255, 0, 0), -1)
                flag, rvec, tvec = cv2.solvePnP(kpts3d, kpts2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                flag, rvecs, tvecs, ret = cv2.solvePnPGeneric(kpts3d[[3,0,1,2]], kpts2d[[3,0,1,2]], cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                rvecs = list(rvecs) + [rvec]
                tvecs = list(tvecs) + [tvec]
                best_R, best_T = None, None
                best_iou, best_mask, best_dist = 0.0, None, 1e6
                ys, xs = np.where(alpha > 0.5)
                centerx, centery = xs.mean(), ys.mean()
                gaussians = GaussianModel2D(3)
                gaussians.load_ply([pc_path])
                for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    R, _ = cv2.Rodrigues(rvec)
                    T = tvec / 1000.0
                    render_mask = np.zeros_like(mask)
                    xyz = gaussians._xyz[0].detach().cpu().numpy() @ R.T + T[:,0]
                    ix = xyz[:,0] / xyz[:,-1] * cameraMatrix[0,0] + cx
                    iy = xyz[:,1] / xyz[:,-1] * cameraMatrix[1,1] + cy
                    valid_uv = (ix >= 0) * (ix < width) * (iy >= 0) * (iy < height)
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
                if best_T is None: 
                    print("no valid pose", brand_str)
                    nopose += 1
                    continue
                
                query = {'R': best_R.tolist(), 'T': best_T[:,0].tolist(), 'gs_path': pc_dir,
                        'track_id': track_id, 'kpts3d': kpts3d.tolist(), 'name': gs_dir}
                queries.append(query)
            if (len(queries) == 0):
                print("no valid query", brand_str)
                noquery += 1
                continue
            render_image[best_mask > 0.5,:] = render_image[best_mask > 0.5,:] * 0.5 + 0.5 * np.array([0, 255, 0])
            query_dict[track_id] = {
                'fovxy': [FovX, FovY],
                'bbox': bbox,
                'xyxy': xyxy,
                'brand': [brand, subbrand, year, color],
                'kpts2d': kpts2d.tolist(),
                'occlude': occlude,
                'queries': queries,}
            valid += 1

        if len(query_dict) == 0:
            print("no valid query", img_path)
            nodict += 1
            continue
        
        # save_path = img_path.replace('trial_v2', 'pnp')
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imwrite(save_path, render_image[...,::-1])

        # json_path = img_path.replace('trial_v2', 'dicts')[:-3] + 'json'
        # os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # with open(json_path, 'w') as f:
        #     json.dump(query_dict, f, indent=4, ensure_ascii=False)
        success += 1
        # break
    print(count, success, nodict, totoal_num)
    print(total, valid, small, no2dgs, no3dkpts, noquery, nomask, nopose, nobrand)

def isinside(alpha):
    height, width = alpha.shape
    x, y, w, h = cv2.boundingRect(alpha)
    if (x <= 0 and alpha[:,x].sum() >= w) or (x+w >= width-1 and alpha[:,x+w-1].sum() >= w) or \
       (y <= 0 and alpha[y,:].sum() >= h) or (y+h >= height-1 and alpha[y+h-1,:].sum() >= h):
           return False
    return True


def search_car_brand(sfm_list, brand, subbrand, color):
    sfm_paths = []
    for sfm_path in sfm_list:
        if brand in sfm_path and subbrand in sfm_path and color in sfm_path:
            image_dir = os.path.join(sfm_path.replace('2dgs', 'sfm'), 'images')
            if not os.path.exists(image_dir):
                continue
            img_paths = os.listdir(image_dir)
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
