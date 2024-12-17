import os
import sys
import cv2
import json
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import shutil
from pdb import set_trace
from plyfile import PlyData, PlyElement

def read_url_image(url):
    response = requests.get(url)
    mask_array = np.array(bytearray(response.content), dtype='uint8')
    mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
    return mask

def process_seg():
    # seg_json = './metaloop_20241126205435/output.json'
    seg_json = './metaloop_20241126210108/output.json'
    with open(seg_json, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]

    index2occluded = {}
    lines = sorted(lines, key=lambda x: os.path.basename(x['image_path']))
    for sample in tqdm(lines):
        results = sample['result']
        if len(results) == 0:
            continue
        if len(results) == 1 and results[0]['tagtype'] == 'delete':
            continue
        image_path = os.path.join(os.path.dirname(seg_json), sample['image_path'])
        image = cv2.imread(image_path, -1)
        height, width, _ = image.shape
        
        basename = os.path.basename(image_path).split('.')[0]
        index2occluded[basename] = []
        instance = np.zeros_like(image[...,0])
        count = 1
        for result in results:
            if result.get('tagtype') == 'delete' or result.get('datatype') not in ['mask', 'polygon']:
                continue
            if result.get('datatype') == 'mask':
                try:
                    mask = read_url_image(result['maskData'])
                except: 
                    continue
            elif result.get('datatype') == 'polygon':
                contours = [np.array(result['data']).astype('int').reshape(1,-1,2)]
                mask = np.zeros((height, width), dtype=np.uint8)
                # 将每个轮廓绘制在掩码图像上
                cv2.drawContours(mask, contours, -1, 255, -1)
                mask = mask.reshape(height, width, 1)

            ret, mask = cv2.threshold(np.max(mask, axis=2), 1, 255, cv2.THRESH_BINARY)
            
            if result.get('tagtype') == 'occlude' or result.get('tagname') == '遮挡车辆_mask':
                index2occluded[basename].append(count)
            instance[mask > 0] = count
            count += 1
        
        if instance.max() == 0:
            print(image_path, 'no instance')
            continue
        save_path = image_path.replace('trial_v1', 'masks')[:-3] + 'png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, instance)
        
        # show_path = os.path.join('show', os.path.basename(image_path))
        # os.makedirs(os.path.dirname(show_path), exist_ok=True)
        # for idx in index2occluded[basename]:
        #     alpha = (instance == idx) * 0.7
        #     image = image * (1-alpha[...,None]) + alpha[...,None]*np.array([0,255,0])[None,None,:]
        # # alpha = instance/instance.max() * 0.7
        # # image = image * (1-alpha[...,None]) + alpha[...,None]*np.array([0,255,0])[None,None,:]
        # cv2.imwrite(show_path, image)
        
    json_path = os.path.join(os.path.dirname(seg_json), 'occluded.json')
    with open(json_path, 'w') as f:
        json.dump(index2occluded, f, indent=4, ensure_ascii=False)
    
        

def process_brand():
    # path = "./infos/crop_brand_v2.json"
    path = "./infos_v1/crop_brand_v1.json"
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    brand_dict = {}
    for sample in lines:
        image_path = sample['image_path']
        basename = os.path.basename(image_path).split('.')[0]
        track_id, frame_id = [x for x in basename.split('_')]
        # if "0000315" in basename:
        #     set_trace()
        if frame_id not in brand_dict:
            brand_dict[frame_id] = {}
        for result in sample['result']:
            if not result or result['datatype'] not in ['extra']:
                continue
            
            for data in result['extraData']:
                if data['tagname'] == 'brand':
                    brand = data['tagvalue']
                elif data['tagname'] =='subbrand':
                    subbrand = data['tagvalue']
                elif data['tagname'] == 'year':
                    year = data['tagvalue']
            
            brand_dict[frame_id][track_id] = [brand, subbrand, year]

    # save_path = './infos/brand_dict_v2.json'
    save_path = './infos_v1/brand_dict_v1.json'
    with open(save_path, 'w') as f:
        json.dump(brand_dict, f, indent=4, ensure_ascii=False)
    
def process_color():
    # path = "./infos/result_color.json"
    # save_path = './infos/color_dict_v2.json'
    path = "./infos_v1/trial_v1_batch1.json"
    save_path = './infos_v1/color_dict_v1.json'

    tagid2color = {
        501001001: '蓝',
        501001002: '灰',
        501001003: '棕',
        501001004: '紫',
        501001005: '黄',
        501001006: '粉',
        501001007: '黑',
        501001008: '橙',
        501001009: '绿',
        501001010: '白',
        501001011: '银',
        501001012: '红',
    }
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    path = "./infos_v1/trial_v1_batch2.json"
    with open(path, 'r') as f:
        lines += [json.loads(line) for line in f.readlines()]

    color_dict = {}
    for sample in tqdm(lines):
        image_path = sample['image']
        basename = os.path.basename(image_path).split('.')[0]
        track_id, frame_id = [x for x in basename.split('_')]
        if frame_id not in color_dict:
            color_dict[frame_id] = {}
        
        confidence, tagid = 0, -1
        for result in sample['result']:
            if result['tagnameid'] not in tagid2color:
                print('tagnameid not in tagid2color', result['tagnameid'])
                continue
            if result['confidence'][0] > confidence:
                confidence = result['confidence'][0]
                tagid = result['tagnameid']
        
        if tagid == -1:
            print('tagid not in valid', tagid)
            continue
        
        color_dict[frame_id][track_id] = tagid2color[tagid]
    
    print(len(color_dict), '0000007' in color_dict)
    with open(save_path, 'w') as f:
        json.dump(color_dict, f, indent=4, ensure_ascii=False)
    
    
def process_trackid():
    path = "./infos/output-9.json"
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    track_dict = {}
    for sample in tqdm(lines):
        image_path = sample['image_path']
        basename = os.path.basename(image_path).split('.')[0]
        track_id, frame_id = [x for x in basename.split('_')]
        if len(sample['result']) == 0:
            print('no result', track_id, frame_id)
            continue
        if len(sample['result']) == 1 and sample['result'][0]['tagtype'] == 'delete':
            print('invalid result', track_id, frame_id, sample['result'][0])
            continue
        if track_id not in track_dict:
            track_dict[track_id] = {}
        
        for result in sample['result'][1]['extraData']:
            if result['tagname'] == 'brand':
                brand = result['tagvalue']
            elif result['tagname'] =='subbrand':
                subbrand = result['tagvalue']
            elif result['tagname'] == 'year':
                year = result['tagvalue']
        
        if "other" in [brand, subbrand, year]:
            print("unkown brand, subbrand, year", track_id, frame_id, brand, subbrand, year)
            continue
        track_dict[track_id][frame_id] = [brand, subbrand, year]
        
    brand_dict = {}
    for track_id in track_dict.keys():
        brands = []
        for frame_id in track_dict[track_id].keys():
            brands.append(tuple(track_dict[track_id][frame_id]))
        
        brands = list(set(brands))
        if track_id == '2':
            brands = [('奥迪', '奥迪A4L', '2013款')]
        if track_id == '106':
            brands = [('大众', '帕萨特', '2016款')]
        if track_id == '103':
            brands = [('斯柯达', '明锐', '2015款')]
        if track_id == '342':
            brands = [('丰田', '威兰达', '2021款')]
        if track_id == '116':
            brands = [('比亚迪', '汉', '2023款')]
        if track_id == '112':
            brands = [('长安跨越', '跨越星V5', '2021款')]
        if track_id == '131':
            brands = [('标致', '标致408', '2019款')]
        if len(brands) > 1: 
            print(track_id, list(track_dict[track_id].keys()), brands)
        for frame_id in track_dict[track_id].keys():
            if frame_id not in brand_dict:
                brand_dict[frame_id] = {}
            brand_dict[frame_id][track_id] = brands[0]
    
    save_path = './infos/brand_dict_v2new.json'
    with open(save_path, 'w') as f:
        json.dump(brand_dict, f, indent=4, ensure_ascii=False)
        

def process_infos():
    # path = './metaloop_20241126205435/metaloop_data/trial_v2/*jpg'
    # bbox_file = './infos/trial_v2_runtime_2024_10_15_output_bbox_5010_car.txt'
    # color_file = './infos/color_dict_v2.json'
    # brand_file = './infos/brand_dict_v2.json'
    # kpts_file = './infos/kps_trial_v2.json'
    # save_path = './infos/infos_dict_v2.json'

    path = './metaloop_20241126210108/metaloop_data/trial_v1/*jpg'
    bbox_file = './infos_v1/trial_v1_runtime_2024_10_15_output_bbox_5010_car.txt'
    color_file = './infos_v1/color_dict_v1.json'
    brand_file = './infos_v1/brand_dict_v1.json'
    kpts_file = './infos_v1/kps_trial_v1_batch1.json'
    save_path = './infos_v1/infos_dict_v1.json'

    image_paths = sorted(glob(path))
    
    bboxes = np.loadtxt(bbox_file, delimiter=',')
    
    bbox_dict = {}
    for bbox in bboxes:
        frame_id, track_id = bbox[:2]
        if frame_id not in bbox_dict:
            bbox_dict[frame_id] = {}
        bbox_dict[frame_id][int(track_id)] = bbox[2:6].tolist()    # xywh
        
    with open(color_file, 'r') as f:
        color_dict = json.load(f)

    with open(brand_file, 'r') as f:
        brand_dict = json.load(f)
    
    with open(kpts_file, 'r') as f:
        kpts_data = json.load(f)
    kpts_file = './infos_v1/kps_trial_v1_batch2.json'
    print(len(kpts_data))
    with open(kpts_file, 'r') as f:
        kpts_data2 = json.load(f)
    kpts_data.update(kpts_data2)
    print(len(kpts_data), len(kpts_data2))
    set_trace()
    kpts_dict = {}
    for key in kpts_data.keys():
        track_id, frame_id = [x for x in key.split('.')[0].split('_')]
        if frame_id not in kpts_dict:
            kpts_dict[frame_id] = {}

        kpts_dict[frame_id][track_id] = kpts_data[key]
    
    infos_dict = {}
    for image_path in tqdm(image_paths):
        basename = os.path.basename(image_path).split('.')[0]
        frame_id = int(basename)
        
        # pose_path = os.path.join('./infos/crop_brand_v2', basename.replace('jpg', 'json'))
        # with open(pose_path, 'r') as f:
        #     pose_data = json.load(f)
        # key = basename.split('.')[0]
        # pose_dict = {k.split('_')[0]: v for k, v in pose_data[key][0].items()}
        if basename not in color_dict:
            print(basename, 'not in color_dict')
            continue
        if basename not in brand_dict:
            print(basename, 'not in brand_dict')
            continue
        if basename not in kpts_dict:
            print(basename, 'not in kpts_dict')
            continue
        
        track_dict= {}
        track_ids = list(kpts_dict[basename].keys())
        for track_id in track_ids:
            color = color_dict[basename][track_id]
            brand = brand_dict[basename].get(track_id, ['null', 'null', 'null'])
            if not brand:
                print(basename, track_id, 'not in brand_dict')
                # continue
            if track_id not in track_dict:
                track_dict[track_id] = {}
            kpts = kpts_dict[basename][track_id]
            bbox = bbox_dict[frame_id][int(track_id)]
            track_dict[track_id]['color'] = color
            track_dict[track_id]['brand'] = brand
            track_dict[track_id]['kpts'] = kpts
            track_dict[track_id]['bbox'] = bbox
        
        infos_dict[image_path] = track_dict
    
    with open(save_path, 'w') as f:
        json.dump(infos_dict, f, indent=4, ensure_ascii=False)

def find_best_intr():
    # path = './infos/infos_dict_v2.json'
    # save_path = './infos/best_list_v2.txt'

    path = './infos_v1/infos_dict_v1.json'
    save_path = './infos_v1/best_list_v1.txt'    
    
    with open(path, 'r') as f:
        infos_dict = json.load(f)
    
    root = './cars/2dgs'
    sfm_list = brand_list(root)


    max_num = 0
    best_list = []
    for image_path in infos_dict.keys():
        track_dict = infos_dict[image_path]
        num = len(track_dict)
        if num < max_num:
            print(image_path, num, '<=', max_num)
            continue
        count = 0
        brands = []
        for track_id in track_dict.keys():
            brand, subbrand, year = track_dict[track_id]['brand']
            color = track_dict[track_id]['color']
            sfm_path = search_car_brand(sfm_list, brand, subbrand, color)
            if not sfm_path:
                print(image_path, track_id, num, brand, color, 'not found in sfm_list')
                continue
            count += 1
            brands.append('_'.join([brand, subbrand, year]))
        
        if count >= max_num:
            max_num = count
            best_list.append([image_path, count, brands])
    
    best_list = sorted(best_list, key=lambda x: x[1], reverse=True)
    print(best_list)
    with open(save_path, 'w') as f:
        f.writelines([str(line) + '\n' for line in best_list])
            

def brand_list(path):
    sfm_paths = []
    for dir in os.listdir(path):
        car_dir = os.path.join(path, dir)
        for color in os.listdir(car_dir):
            sparse_dir = os.path.join(car_dir, color)
            image_dir = os.path.join(sparse_dir, 'images/*.jpg')
            # image_list = glob(image_dir)
            # num = len(image_list)
            # if num > 50:
            #     sfm_paths.append(sparse_dir)
            sfm_paths.append(sparse_dir)
    
    return sfm_paths

def search_car_brand(sfm_list, brand, subbrand, color):
    if not brand:
        return None
    for sfm_path in sfm_list:
        if brand in sfm_path and color in sfm_path: # and subbrand in sfm_path:
            return sfm_path
    return None

def process_kpts3d():
    path = "./infos/car_param.json"
    with open(path, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]
        
    kpts3d_dict = {}
    for sample in tqdm(lines):
        carname = sample['name']
        width = int(sample.get('宽度(mm)', 0))
        wheelbase = int(sample.get('轴距(mm)', 0))
        front_wheel_distance = int(sample.get('前轮距(mm)', 0))
        rear_wheel_distance = int(sample.get('后轮距(mm)', 0))
        
        if front_wheel_distance > rear_wheel_distance:
            front_width = width
            rear_width = width - (front_wheel_distance - rear_wheel_distance)
        elif rear_wheel_distance > front_wheel_distance:
            rear_width = width
            front_width = width - (rear_wheel_distance - front_wheel_distance)
        else:
            front_width = width
            rear_width = width
        points_3d = np.array([
            [wheelbase/2, front_width/2, 0],
            [wheelbase/2, -front_width/2, 0],    
            [-wheelbase/2, -rear_width/2, 0],    
            [-wheelbase/2, rear_width/2, 0]    
        ])
        kpts3d_dict[carname] = points_3d.tolist()
    
    save_path = './infos/kpts3d_dict.json'
    with open(save_path, 'w') as f:
        json.dump(kpts3d_dict, f, indent=4, ensure_ascii=False)
        

def search_best_intr():
    path = './pose/delta_pose_0012*v2.json'
    json_list = sorted(glob(path))
    
    losses = []
    counts = []
    intris = []
    for json_path in json_list:
        with open(json_path, 'r') as f:
            data = json.load(f)
        loss = data['best_loss']
        RT = data['best_RT']
        intr = data['intr']
        
        losses.append(loss)
        counts.append(len(RT))
        intris.append(intr)
    
    losses = np.array(losses)
    counts = np.array(counts)
    intris = np.array(intris)
    index = np.argsort(losses)
    losses3 = losses[index][counts[index] == 3]
    intris3 = intris[index][counts[index] == 3]
    json_list3 = [json_list[i] for i in index if counts[i] == 3]
    
    losses4 = losses[index][counts[index] == 4]
    intris4 = intris[index][counts[index] == 4]
    json_list4 = [json_list[i] for i in index if counts[i] == 4]
    set_trace()


def remove_logs2():
    path = './logs2/*log'
    log_paths = sorted(glob(path))
    
    poses_path = "metaloop_20241126205435/metaloop_data/poses_v2"
    tgt_path = poses_path.replace('v2', 'v2_new')
    os.makedirs(tgt_path, exist_ok=True)
    for log_path in tqdm(log_paths):
        # with open(log_path, 'r') as f:
        #     text = f.read()
        
        # if "no valid query" in text:
        #     os.remove(log_path)
        basename = os.path.basename(log_path).split('.')[0]
        fname = basename[:-2]
        pose_paths = sorted(glob(os.path.join(poses_path, fname + '*')))
        for pose_path in pose_paths:
            shutil.move(pose_path, tgt_path)
        
        

def merge_proses():
    path = "metaloop_20241126205435/metaloop_data"
    dirs = ['poses_v1', 'poses_v2_old', 'poses_v3', 'poses_v4']
    img_paths = sorted(glob(os.path.join(path, 'trial_v2/*jpg')))
    for img_path in img_paths:
        basename = os.path.basename(img_path).split('.')[0] + '.json'
        tgt_path = os.path.join(path, 'poses', basename)
        os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
        if os.path.exists(tgt_path):
            with open(tgt_path, 'r') as f:
                data_dict = json.load(f)
        else:
            data_dict = {}
        for pose_dir in dirs:
            src_path = os.path.join(path, pose_dir, basename)
            if not os.path.exists(src_path):
                continue
            with open(src_path, 'r') as f:
                try:
                    new_dict = json.load(f)
                except:set_trace()
            
            for track_id in new_dict.keys():
                if track_id not in data_dict:
                    data_dict[track_id] = new_dict[track_id]
                else:
                    for kkey in new_dict[track_id].keys():
                        if not isinstance(new_dict[track_id][kkey]['pose'][0], list):
                            continue
                        if kkey not in data_dict[track_id]:
                            data_dict[track_id][kkey] = new_dict[track_id][kkey]
                            img_path = src_path[:-5] + '%s.png' % kkey
                            imgname = os.path.basename(img_path)
                            if os.path.exists(img_path):
                                shutil.move(img_path, os.path.join(path, 'poses', imgname))
                        elif isinstance(new_dict[track_id][kkey]['loss'], list):
                            if isinstance(data_dict[track_id][kkey]['loss'], list):
                                data_loss = data_dict[track_id][kkey]['loss'][0]
                            else:
                                data_loss = data_dict[track_id][kkey]['loss']
                                
                            if new_dict[track_id][kkey]['loss'][0] < data_loss:
                                data_dict[track_id][kkey] = new_dict[track_id][kkey]
                                img_path = src_path[:-5] + '%s.png' % kkey
                                imgname = os.path.basename(img_path)
                                if os.path.exists(img_path):
                                    shutil.move(img_path, os.path.join(path, 'poses', imgname))
                        else:
                            if isinstance(data_dict[track_id][kkey]['loss'], list):
                                data_loss = data_dict[track_id][kkey]['loss'][0]
                            else:
                                data_loss = data_dict[track_id][kkey]['loss']

                            if new_dict[track_id][kkey]['loss'] < data_loss:
                                data_dict[track_id][kkey] = new_dict[track_id][kkey]
                                img_path = src_path[:-5] + '%s.png' % kkey
                                imgname = os.path.basename(img_path)
                                if os.path.exists(img_path):
                                    shutil.move(img_path, os.path.join(path, 'poses', imgname))
                        
        with open(tgt_path, 'w') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
            

def find_nonconvex_contour():
    # 读取图像
    mask = cv2.imread('zmask2.jpg', 0)
    image = cv2.imread('zimg.jpg', -1)
    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓（假设车辆轮廓是最大的）
    max_contour = max(contours, key=cv2.contourArea)

    # 计算凸包
    hull = cv2.convexHull(max_contour, returnPoints=False)

    # 检测凹凸缺陷
    defects = cv2.convexityDefects(max_contour, hull)

    max_len = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        
        # length = np.linalg.norm(np.array(start) - np.array(end))
        length = (e - s) if e > s else (e + len(max_contour) - s)
        
        if length > max_len:
            max_len = length
            start0 = tuple(max_contour[s][0])
            end0 = tuple(max_contour[e][0])
            far0 = tuple(max_contour[f][0])
            print(start, end, far, defects[i, 0], length)
            
        # 绘制缺陷
    cv2.line(image, start0, end0, [0, 255, 0], 2)
    cv2.circle(image, far0, 1, [0, 0, 255], -1)

    cv2.imwrite("zcnt.jpg", image)

def find_inner_contour():
    mask = cv2.imread('zmask2.jpg', 0)
    image = cv2.imread('zimg.jpg', -1)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
def remove_2dgs_pc():
    root = './cars/2dgs'
    paths = []
    for dir in os.listdir(root):
        car_dir = os.path.join(root, dir)
        for color in os.listdir(car_dir):
            ply_path = os.path.join(car_dir, color, 'point_cloud/iteration_30000/point_cloud.ply')
            if not os.path.exists(ply_path):
                continue
            paths.append(ply_path)
    
    for path in tqdm(paths):
        remove_pc(path)

def remove_pc(path):
    print(path)
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    
    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    mask = xyz[:,-1] > 0.1
    xyz = xyz[mask]
    opacities = opacities[mask]
    features_dc = features_dc[mask]
    features_extra = features_extra[mask]
    scales = scales[mask]
    rots = rots[mask]
    
    construct_list_of_attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(features_dc.shape[1]):
        construct_list_of_attributes.append('f_dc_%d' % i)
    for i in range(features_extra.shape[1]):
        construct_list_of_attributes.append('f_rest_%d' % i)
    construct_list_of_attributes.append('opacity')
    for i in range(scales.shape[1]):
        construct_list_of_attributes.append('scale_%d' % i)
    for i in range(rots.shape[1]):
        construct_list_of_attributes.append('rot_%d' % i)
    
    normals = np.zeros_like(xyz)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, features_dc, features_extra, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    save_path = path.replace("30000", "7000")
    PlyData([el]).write(save_path)

    
def crop_images():
    path = "metaloop_20241126205435/metaloop_data/poses_v3/*png"
    img_paths = sorted(glob(path))
    height, width = 1520, 2688
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        if (height, width) != img.shape[:2]:
            shutil.copy(img_path, img_path.replace("poses_v3", "crop"))
            continue
        # render_path = img_path.replace("poses_v3", "render")
        # render = cv2.imread(render_path, -1)
        # rgba_path = img_path.replace("poses_v3", "rgba")
        # rgba = cv2.imread(rgba_path, -1)
        # mask = (render[..., 3] + rgba[..., 3] > 0)
        # ys, xs = np.where(mask > 0)
        # x0, y0 = xs.min(), ys.min()
        # x1, y1 = xs.max(), ys.max()
        # x0 = max(0, x0 - 10)
        # x1 = min(width-1, x1 + 10)
        # y0 = max(0, y0 - 10)
        # y1 = min(height-1, y1 + 10)
        # crop = img[y0:y1, x0:x1]
        # save_path = img_path.replace("poses_v3", "crop")
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # cv2.imwrite(save_path, crop)


if __name__ == '__main__':
    # process_seg()
    # process_brand()
    # process_color()
    # process_trackid()
    # process_infos()
    # find_best_intr()
    # process_kpts3d()
    # search_best_intr()
    # remove_logs2()
    # merge_proses()
    # find_nonconvex_contour()
    # find_inner_contour()
    # remove_2dgs_pc()
    crop_images()
