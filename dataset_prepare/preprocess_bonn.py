# %%
import sys
import os

import glob
import os
import shutil
import numpy as np
from evo.tools import file_interface
from util import convert_trajectory_to_extrinsic_matrices
from tqdm import tqdm

# Camera intrinsic parameters
fx = 542.822841
fy = 542.576870
cx = 315.593520
cy = 237.756098

# Original intrinsic matrix
K_original = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

dirs = glob.glob("/home/wkr/workspace/workspace_yuzhi/uni4d/dataset_prepare/bonn/rgbd_bonn_dataset/*/")
dirs = sorted(dirs)

# extract frames
for dir in tqdm(dirs):
    out_dir = f"./data/bonn/{dir.split('/')[-2]}"
    frames = glob.glob(dir + 'rgb/*.png')
    frames = sorted(frames)
    # sample 110 frames at the stride of 2
    frames = frames[30:140]
    # cut frames after 110
    new_dir = os.path.join(out_dir, "rgb")

    for frame in frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)

    depth_frames = glob.glob(dir + 'depth/*.png')
    depth_frames = sorted(depth_frames)
    depth_frames = depth_frames[30:140]
    new_dir = os.path.join(out_dir, "depth_gt")

    for frame in depth_frames:
        os.makedirs(new_dir, exist_ok=True)
        shutil.copy(frame, new_dir)

    gt_path = "groundtruth.txt"
    traj = file_interface.read_tum_trajectory_file(dir + gt_path)
    t_gt = traj.positions_xyz[30:140]
    R_gt = traj.orientations_quat_wxyz[30:140]
    c2w = convert_trajectory_to_extrinsic_matrices(t_gt, R_gt)

    intrinsics = np.repeat(K_original[np.newaxis, :, :], len(c2w), axis=0)
    
    np.save(os.path.join(out_dir, "K.npy"), intrinsics)
    np.save(os.path.join(out_dir, "c2w.npy"), c2w)



