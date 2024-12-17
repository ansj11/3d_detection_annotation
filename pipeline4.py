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
from multiprocessing import Process, Lock
from pdb import set_trace
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='metaloop_20241126205435/metaloop_data/dicts/*json',
                        help="input path")
args = parser.parse_args()


def main():
    root = args.input
    
    image_paths = sorted(glob(root))
    # filter_paths = []
    # for image_path in image_paths:
    #     json_path = image_path.replace('dicts', 'poses_v3')
    #     if os.path.exists(json_path):
    #         continue
    #     filter_paths.append(image_path)
    
    # print("chosen images:", len(image_paths), len(filter_paths))
    # image_paths = filter_paths
        
    n_gpu = torch.cuda.device_count()
    total_num = 6 * n_gpu
    length = len(image_paths)
    interval = int(length/total_num) + 1
    pool = []
    for i in range(total_num):
        start = i * interval
        end = min((i+1) * interval, length)
        thread = Process(target=func, args=(image_paths[start:end], i), name=f"thread_{i}")
        pool.append(thread)
        
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    
    print('parallel pose optimize done!')

def func(paths, idx):
    for i, path in enumerate(paths):
        
        n_gpu = torch.cuda.device_count()
        gpu_id = idx % n_gpu
        path = path.replace('(', '\(').replace(')', '\)').replace('|', '\|')
        basename = os.path.basename(path)[:-4]
        print('traing %d: %d/%d ' % (idx, i, len(paths)), path)
        cmd_str = "CUDA_VISIBLE_DEVICES=%d nohup python -u process4.py --input %s > logs4/%s.log" %(gpu_id, path, basename)
        os.system(cmd_str)


if __name__ == "__main__":
    main()
