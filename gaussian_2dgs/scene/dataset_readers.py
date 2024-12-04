#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import sys
from PIL import Image
from typing import NamedTuple
from .colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from glob import glob
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from .gaussian_model import BasicPointCloud
from pdb import set_trace

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    valid_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)    # mean t
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)     # 6.809671
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)  # t=[0.83072447, 0.42330418, 4.72019668]
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])    # -R*t or =t*R'

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1 # 7.49

    translate = -center # use camera translation mean as center

    return {"translate": translate, "radius": radius}   # camera position radius

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num = len(glob(os.path.join(images_folder, '*jpg')))
    if len(cam_extrinsics) != num:
        print("pose num != image num: ", len(cam_extrinsics), num)
        raise Exception
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))    # qvec to rotmat
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)    # read ply comments
    vertices = plydata['vertex']    # vertex attribution
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T  # xyz:(182686, 3)
    # rescale the sfm point cloud
    print(positions)

    positions = positions / 1000.0
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0    # rgb
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T     # normals: [0,0,0]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    # try:
    #     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")   # read colmap
    #     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    #     cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file) # {1~301:{id=1,qvec,tvec,camera_id,name,xyz,point3D_ids}}
    #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file) # {1:{id,model='PINHOLE',width=1959, height=1090, params=array([1159.5880733,1164.66012875,979.5,545}}
    # except:
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)    # [CameraInfo(uid=1,...)]

    cam_centers = []
    for i, cam in enumerate(cam_infos):
        # y = x*R+T
        R = cam.R
        T = cam.T
        cam_center = -np.matmul(R, T)
        cam_centers.append(cam_center)
    cam_center = np.stack(cam_centers, axis=0)
    X = np.stack([cam_center[:,0], cam_center[:,2], np.ones_like(cam_center[:,0])], axis=-1)
    Y = cam_center[:,1]
    XTX = np.matmul(X.transpose(1, 0), X)
    XTY = np.matmul(X.transpose(1, 0), Y)
    A = np.matmul(np.linalg.inv(XTX), XTY)
    
    y = A[0] * cam_center[:,0] + A[1] * cam_center[:,2] + A[2]

    test_set_indices = np.where(y >= Y)[0]
    training_set_indices = np.where(y < Y)[0]
    # for debug 
    # concat traiing indices with test indices
    training_set_indices = np.concatenate((training_set_indices, test_set_indices))
    test_set_indices = []

    # path0 = "data/jietuV9/images"
    # path1 = "data/jietuV5/images"
    # fnames = os.listdir(path0)
    # fnames1 = os.listdir(path1)
    # training_set_indices = [int(x.split('.')[0])-1 for x in fnames]
    # test_set_indices = [int(x.split('.')[0])-1 for x in fnames1 if x not in fnames]
    print('training_set_indices: ', training_set_indices)
    print('test_set_indices: ', test_set_indices)
    
    train_cam_infos = [cam_infos[i] for i in training_set_indices]
    valid_cam_infos = []
    test_cam_infos = [cam_infos[i] for i in test_set_indices]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not os.path.exists(ply_path):    # False
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
    #     xyz, rgb = filterPointCloud(xyz, rgb, cam_infos)
    #     storePly(ply_path, xyz, rgb)
    print(f"txt path: {txt_path}")
    xyz, rgb, _ = read_points3D_text(txt_path)
    xyz, rgb = filterPointCloud(xyz, rgb, cam_infos)
    print(xyz)
    print("filter point cloud done")
    if os.path.exists(ply_path):
        os.remove(ply_path)
    storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)    # BasicPointCloud class
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           valid_cameras=valid_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,   # {'translate': array([ 0.09687055, -0.0013478 ,  0.00561222], dtype=float32), 'radius': 7.49}
                           ply_path=ply_path)
    return scene_info

def filterPointCloud(points, colors, cam_infos):
    print("filter point cloud with mask, ", points.shape)
    for cam in tqdm(cam_infos):
        R, T = cam.R, cam.T
        mask = np.array(cam.image)[...,-1]
        trans_points = np.matmul(points, R) + T
        fx, fy = cam.width / 2 / np.tan(cam.FovX/2), cam.height / 2 / np.tan(cam.FovY/2)
        ix = trans_points[:,0] / trans_points[:,-1] * fx + cam.width / 2
        iy = trans_points[:,1] / trans_points[:,-1] * fy + cam.height / 2
        valid_uv = (ix >= 0) * (ix < cam.width) * (iy >= 0) * (iy < cam.height)
        points = points[valid_uv]
        colors = colors[valid_uv]
        ix, iy = ix[valid_uv].astype('int'), iy[valid_uv].astype('int')
        filter_points, filter_colors = [], []
        threshold = mask.max() / 2
        for idx in range(len(points)):
            x, y = ix[idx], iy[idx]
            if mask[y,x] > threshold:
                filter_points.append(points[idx])
                filter_colors.append(points[idx])
        points = np.stack(filter_points, axis=0)
        colors = np.stack(filter_colors, axis=0)
        if False:
            image = np.array(cam.image)[...,:3]
            render_image = np.zeros_like(image)
            render_image[iy, ix] = colors
            cv2.imwrite('debug/%s.jpg' % cam.image_name, render_image[...,::-1])
    print("filtered point cloud num, ", points.shape)
    return points, colors

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}