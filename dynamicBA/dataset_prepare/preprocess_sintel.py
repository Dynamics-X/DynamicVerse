#!/usr/bin/env python3

import os
import shutil
import numpy as np
from sintel_io import cam_read, depth_read
from tqdm import tqdm

# Paths
source_dir = "./dataset_prepare/sintel/training"
target_dir = "./data/sintel"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get a list of all video dirs (assuming they're the same in all subdirs)
video_dirs = sorted(os.listdir(os.path.join(source_dir, "final")))

# Process each video
for video_name in tqdm(video_dirs):
    
    # Create directories for this video
    video_target_dir = os.path.join(target_dir, video_name)
    rgb_target_dir = os.path.join(video_target_dir, "rgb")
    occ_target_dir = os.path.join(video_target_dir, "occ")
    depth_target_dir = os.path.join(video_target_dir, "depth_gt")
    
    os.makedirs(rgb_target_dir, exist_ok=True)
    os.makedirs(depth_target_dir, exist_ok=True)
    os.makedirs(occ_target_dir, exist_ok=True)
    
    # 1. Copy RGB images
    rgb_source_dir = os.path.join(source_dir, "final", video_name)
    for img_file in os.listdir(rgb_source_dir):
        if img_file.endswith((".png")):
            shutil.copy(
                os.path.join(rgb_source_dir, img_file),
                os.path.join(rgb_target_dir, img_file)
            )
    
    # 2. Process and copy depth images
    depth_source_dir = os.path.join(source_dir, "depth", video_name)
    for depth_file in os.listdir(depth_source_dir):
        if depth_file.endswith(".dpt"):
            # Read depth file
            depth_path = os.path.join(depth_source_dir, depth_file)
            depth_data = depth_read(depth_path)
            
            # Save as numpy file
            output_name = os.path.splitext(depth_file)[0] + ".npy"
            np.save(os.path.join(depth_target_dir, output_name), depth_data)
    
    # 3. Process camera data
    cam_source_dir = os.path.join(source_dir, "camdata_left", video_name)
    
    # Assuming there's a camera file for each frame with same naming convention
    cam_files = sorted([f for f in os.listdir(cam_source_dir) if f.endswith(".cam")])
    
    # Create arrays to store all intrinsics and extrinsics
    num_frames = len(cam_files)
    intrinsics = []
    extrinsics = np.zeros((num_frames, 4, 4))
    
    for i, cam_file in enumerate(sorted(cam_files)):
        # Read camera data
        cam_path = os.path.join(cam_source_dir, cam_file)
        K, E = cam_read(cam_path)  # K is 3x3 intrinsic, E is 3x4 extrinsic
        
        # Store intrinsic (should be the same for all frames in a video)
        intrinsics.append(K)
        
        # Convert 3x4 extrinsic to 4x4 matrix (add row [0,0,0,1])
        E_4x4 = np.vstack((E, np.array([0, 0, 0, 1])))
        extrinsics[i] = E_4x4
    
    # Use the first intrinsic (they should all be the same)
    K = np.array(intrinsics)
    c2w = np.linalg.inv(extrinsics)
    
    # Save camera data
    np.save(os.path.join(video_target_dir, "K.npy"), K)
    np.save(os.path.join(video_target_dir, "c2w.npy"), c2w)
    
print("Processing complete!")