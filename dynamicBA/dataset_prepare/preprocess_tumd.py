#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images.

For this purpose, you can use the ''associate.py'' script. It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""
import sys
import os
import argparse
import sys
import os
import numpy as np
import shutil

from util import convert_trajectory_to_extrinsic_matrices

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)

    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def find_closest_index(L,t):
    """
    Find the index of the closest value in a list.
    
    Input:
    L -- the list
    t -- value to be found
    
    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end+beginning)/2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best

def read_trajectory(filename, matrix=False):
    """
    Read a trajectory from a text file. 
    
    Input:
    filename -- file to be read
    matrix -- convert poses to 4x4 matrices
    
    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]

    list_ok = []
    for i,l in enumerate(list):
        if l[4:8]==[0,0,0,0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v): 
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n"%(i,filename))
            continue
        list_ok.append(l)
    if matrix :
      traj = dict([(l[0],transform44(l[0:])) for l in list_ok])
    else:
      traj = dict([(l[0],l[1:8]) for l in list_ok])
    return traj



def read_file_list(filename, start=0, end=9999, every=1):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    with open(filename) as f:
        data = f.read()
        lines = data.replace(","," ").replace("\t"," ").split("\n") 
        list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
        list = sorted(list, key=lambda x: float(x[0]))
        list = list[::every][start:end]# takes from start to end
        list = [(l[0],l[1:]) for l in list if len(l)>1]

    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    potential_matches = []
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    for a in first_keys:
        for b in second_keys:
            if abs(float(a) - (float(b) + offset)) < max_difference:
                potential_matches.append((abs(float(a) - (float(b) + offset)), a, b))

    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y

# Original intrinsic matrix
K_original = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
    
parser = argparse.ArgumentParser(description='''
This script takes two data files with timestamps and associates them   
''')

parser.add_argument('--data_dir', default="./dataset_prepare/tum", help='directory containing the data files')
parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
args = parser.parse_args()

videos = sorted(os.listdir(args.data_dir))

for video in videos:

    rgb_file_txt = os.path.join(args.data_dir,video,"rgb.txt")
    depth_file_txt = os.path.join(args.data_dir,video,"depth.txt")
    gt_file_txt = os.path.join(args.data_dir,video,"groundtruth.txt")

    num_files = len(read_file_list(rgb_file_txt, every=1))

    base_output = f"./data/tumd/{video}/"

    start = 0
    end = 90

    rgb_list = read_file_list(rgb_file_txt, start, end, every=3)
    gt_list = read_trajectory(gt_file_txt)

    sorted_rgb_times = sorted(rgb_list.keys(), key=lambda x: float(x))

    stamps_gt = sorted(list(gt_list.keys()))
    gt_poses = []

    output_video = video

    rgb_aligned = os.path.join(base_output,"rgb")
    os.makedirs(rgb_aligned,exist_ok=True)

    for j, t in enumerate(sorted_rgb_times):
        rgb_file = os.path.join(args.data_dir,video, rgb_list[t][0])

        t_gt = stamps_gt[find_closest_index(stamps_gt, float(t))]
        gt_pose = gt_list[t_gt]
        gt_poses.append(gt_pose)

        rgb_file_aligned = os.path.join(rgb_aligned, rgb_list[t][0].split("/")[-1])

        shutil.copy(rgb_file, rgb_file_aligned)

    gt_poses = np.array(gt_poses)
    t_gt = gt_poses[:,:3]
    R_gt = gt_poses[:,3:]
    c2w = convert_trajectory_to_extrinsic_matrices(t_gt, R_gt)

    intrinsics = np.repeat(K_original[np.newaxis, :, :], len(c2w), axis=0)

    np.save(os.path.join(base_output, "K.npy"), intrinsics)
    np.save(os.path.join(base_output, "c2w.npy"), c2w)
