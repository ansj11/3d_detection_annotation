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


def read_url_image(url):
    response = requests.get(url)
    mask_array = np.array(bytearray(response.content), dtype='uint8')
    mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
    return mask

def process_seg():
    seg_json = './metaloop_20241126205435/output.json'
    with open(seg_json, 'r') as f:
        lines = [json.loads(line) for line in f.readlines()]

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
            
            instance[mask > 0] = count
            count += 1
        
        if instance.max() == 0:
            print(image_path, 'no instance')
            continue
        save_path = image_path.replace('trial_v2', 'masks')[:-3] + 'png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, instance)
        
        show_path = os.path.join('show', os.path.basename(image_path))
        os.makedirs(os.path.dirname(show_path), exist_ok=True)
        alpha = instance/instance.max() * 0.7
        show = image * (1-alpha[...,None]) + alpha[...,None]*np.array([0,255,0])[None,None,:]
        cv2.imwrite(show_path, show)
        

def process_brand():
    path = "./infos/crop_brand_v2.json"
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

    save_path = './infos/brand_dict_v2.json'
    with open(save_path, 'w') as f:
        json.dump(brand_dict, f, indent=4, ensure_ascii=False)
    
def process_color():
    path = "./infos/result_color.json"
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
    save_path = './infos/color_dict_v2.json'
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
    path = './metaloop_20241126205435/metaloop_data/trial_v2/*jpg'
    image_paths = sorted(glob(path))
    
    bbox_file = './infos/trial_v2_runtime_2024_10_15_output_bbox_5010_car.txt'
    bboxes = np.loadtxt(bbox_file, delimiter=',')
    
    bbox_dict = {}
    for bbox in bboxes:
        frame_id, track_id = bbox[:2]
        if frame_id not in bbox_dict:
            bbox_dict[frame_id] = {}
        bbox_dict[frame_id][int(track_id)] = bbox[2:6].tolist()    # xywh
        
    color_file = './infos/color_dict_v2.json'
    with open(color_file, 'r') as f:
        color_dict = json.load(f)

    brand_file = './infos/brand_dict_v2.json'
    with open(brand_file, 'r') as f:
        brand_dict = json.load(f)
    
    brand_file = './infos/kps_trial_v2.json'
    with open(brand_file, 'r') as f:
        kpts_data = json.load(f)
    
    kpts_dict = {}
    for key in kpts_data.keys():
        track_id, frame_id = [x for x in key.split('_')]
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
        # infos_dict[image_path] = {
        #     'frame_id': frame_id,
        #     # 'pose': pose_dict,
        #     'mask_path': image_path.replace('trial_v2', 'masks').replace('jpg', 'png'),
        #     'bbox': bbox_dict[frame_id],
        #     'color': color_dict[basename],
        #     'brand': brand_dict[basename],
        #     'kpts': kpts_dict[basename],
        #     "tracks": track_dict,
        # }
    
    save_path = './infos/infos_dict_v2.json'
    # with open(save_path, 'w') as f:
    #     json.dump(infos_dict, f, indent=4, ensure_ascii=False)

def find_best_intr():
    path = './infos/infos_dict_v2.json'
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
    save_path = './infos/best_list_v2.txt'
    with open(save_path, 'w') as f:
        f.writelines([str(line) + '\n' for line in best_list])
            

def brand_list(path):
    sfm_paths = []
    for dir in os.listdir(path):
        car_dir = os.path.join(path, dir)
        for color in os.listdir(car_dir):
            sparse_dir = os.path.join(car_dir, color)
            # image_dir = os.path.join(sparse_dir, 'images/*.jpg')
            # image_list = glob(image_dir)
            # num = len(image_list)
            # if num > 50:
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
        
        

# def move_poses():
#     path = ""
#             os.move(log_path, )
    

if __name__ == '__main__':
    # process_seg()
    # process_brand()
    # process_color()
    # process_trackid()
    # process_infos()
    # find_best_intr()
    # process_kpts3d()
    # search_best_intr()
    remove_logs2()