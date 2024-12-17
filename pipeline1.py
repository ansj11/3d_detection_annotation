import os
import cv2
import json
import argparse
import threading
import subprocess
import numpy as np
from tqdm import tqdm
import torch
from glob import glob
from multiprocessing import Process, Pool
from pdb import set_trace
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='./metaloop_20241126210108/metaloop_data/trial_v1/*jpg',
                        help="input path")
args = parser.parse_args()


def main():
    root = args.input
    
    image_paths = sorted(glob(root))
    filter_paths = []
    for image_path in image_paths:
        json_path = image_path.replace('trial_v1', 'poses')[:-3] + 'json'
        if os.path.exists(json_path):
             continue
        filter_paths.append(image_path)
    
    
    print("chosen images:", len(image_paths), len(filter_paths))
    image_paths = filter_paths
    
    n_gpu = torch.cuda.device_count()
    total_num = 8 * n_gpu
    length = len(image_paths)
    interval = int(length/total_num) + 1

    pool = Pool(processes=total_num)
    for i in range(total_num):
        start = i * interval
        end = min((i+1) * interval, length)
        pool.apply_async(func, (image_paths[start:end], i))
        # thread = Process(target=func, args=(image_paths[start:end], i), name=f"thread_{i}")
        # pool.append(thread)
    
    pool.close()
    pool.join()
    
    # for thread in pool:
    #     thread.start()
    # for thread in pool:
    #     thread.join()
    
    print('parallel pose optimize done!')

def func(paths, idx):
    for i, path in enumerate(paths[::-1]):
        
        n_gpu = torch.cuda.device_count()
        gpu_id = idx % n_gpu
        path = path.replace('(', '\(').replace(')', '\)').replace('|', '\|')
        basename = os.path.basename(path)[:-4]
        print('traing %d: %d/%d ' % (idx, i, len(paths)), path)
        cmd_str = "CUDA_VISIBLE_DEVICES=%d nohup python -u process1.py --input %s > logs/%s.log" %(gpu_id, path, basename)
        # os.system(cmd_str)
        result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 检查命令是否成功执行
        if result.returncode == 0:
            print("Command executed successfully")
        else:
            print("Command failed with return code", result.returncode)


if __name__ == "__main__":
    main()
