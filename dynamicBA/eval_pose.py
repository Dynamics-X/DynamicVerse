import sys
import os
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import json
import torch
import numpy as np
import cv2

from scipy.spatial.transform import Rotation as Rot
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation, Unit
from evo.core import sync
from evo.tools import plot

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_preds(video_path, model, experiment_name, number_of_frames):
    """Get camera pose predictions in camera2world format"""
    images_path = os.path.join(video_path, model, experiment_name)

    if not os.path.exists(images_path):
        print(f"Path {images_path} does not exist")
        return None, None, None         
    
    R_preds = []
    t_preds = [] 

    for i in range(number_of_frames):
        # npz = np.load(os.path.join(images_path, f'{i:04d}.npz'), allow_pickle=True)
        # npz = np.load(os.path.join(images_path, f'{i:04d}_dynamic_flow_fused_wcotracker_noweighted_20epochs_1e-2.npz'), allow_pickle=True)
        # npz = np.load(os.path.join(images_path, f'{i:04d}_before_flow_opt.npz'), allow_pickle=True)
        npz = np.load(os.path.join(images_path, f'{i:04d}_simple_flow_opt_1.0_1e-4.npz'), allow_pickle=True)
        R_preds.append(Rot.from_matrix(npz['R']).as_quat(scalar_first=True)) # convert to quatenion
        t_preds.append(npz['t'])
    
    R_preds, t_preds = np.stack(R_preds, axis=0), np.stack(t_preds, axis=0)

    return R_preds, t_preds


def get_gt(video_path):
    """Load ground truth camera poses in camera2world format"""
    gt_poses = np.load(video_path + '/c2w.npy')
    number_of_frames = gt_poses.shape[0]
    
    # Extract all rotation matrices at once (first 3x3 of each 4x4 matrix)
    rotation_matrices = gt_poses[:, :3, :3]
    
    # Convert all rotation matrices to quaternions in one operation
    # scipy returns quaternions in (x,y,z,w) format
    quats = R.from_matrix(rotation_matrices).as_quat()
    
    # Reorder from scipy's (x,y,z,w) to desired (w,x,y,z) format
    R_gt = np.column_stack((quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]))
    
    # Extract all translations at once (last column of each matrix)
    t_gt = gt_poses[:, :3, 3]

    return R_gt, t_gt, number_of_frames


def evaluate_pose(preds, R_gt, t_gt, normalize=False, video=None, save_path=None):
    """Evaluate camera pose predictions against ground truth"""
    traj_est = PoseTrajectory3D(
        positions_xyz=preds[1],
        orientations_quat_wxyz=preds[0],
        timestamps=np.arange(preds[0].shape[0], dtype=np.float64))

    traj_ref = PoseTrajectory3D(
        positions_xyz=t_gt,
        orientations_quat_wxyz=R_gt,
        timestamps=np.arange(t_gt.shape[0], dtype=np.float64))

    gt_path_length = traj_ref.path_length 

    if normalize:
        traj_ref = PoseTrajectory3D(
            positions_xyz=t_gt / gt_path_length, # normalize to path length 1m
            orientations_quat_wxyz=R_gt,
            timestamps=np.arange(t_gt.shape[0], dtype=np.float64))

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    
    try:
        traj_est.align(traj_ref, correct_scale=True)
        ATE = main_ape.ape(traj_ref, traj_est, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
        RPEt = main_rpe.rpe(traj_ref, traj_est, est_name='traj', align=True, correct_scale=True,
            pose_relation=PoseRelation.translation_part, delta=1, delta_unit=Unit.frames, all_pairs=True)
        RPEr = main_rpe.rpe(traj_ref, traj_est, est_name='traj', align=True, correct_scale=True,
            pose_relation=PoseRelation.rotation_angle_deg, delta=1, delta_unit=Unit.frames, all_pairs=True)
        
        if save_path is not None and video is not None:
            fig = plt.figure()
            traj_by_label = {
                "est": traj_est,
                "reference": traj_ref
            }
            plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            plt.savefig(f"{save_path}/{video}_pose.png")
            plt.close(fig)

        return ATE.stats['rmse'], RPEt.stats['rmse'], RPEr.stats['rmse']
    
    except Exception as e:
        print(e)
        return None, None, None


def eval_pose(dataset="sintel", base_path=None, experiment_name="base", 
          model="dynamicBA", normalize=False):
    """Main function to evaluate camera pose predictions"""
    
    if base_path is None:
        base_path = f"./data/{dataset}"
    
    if normalize:
        save_name = "normalized"
    else:
        save_name = "unnormalized"
    
    if "sintel" in dataset:
        videos = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", 
                 "market_2", "market_5", "market_6", "shaman_3", "sleeping_1", 
                 "sleeping_2", "temple_2", "temple_3"]
    else:
        videos = sorted(os.listdir(base_path))
    
    res_save_path = f"results/pose/{dataset}/{experiment_name}_{save_name}"
    print(res_save_path)
    os.makedirs(res_save_path, exist_ok=True)
    
    # Open a txt file to write results for each video with better formatting
    results_file = open(f"{res_save_path}/all_results.txt", "w")
    
    # Column headers with fixed width formatting
    results_file.write(f"{'Video':<15}{'ATE':<12}{'RPEt':<12}{'RPEr':<12}\n")
    results_file.write("-" * 50 + "\n")  # Separator line

    errs = {
        "ATEs": 0,
        "RPEts": 0,
        "RPErs": 0,
        "pose_err": 0,  # total videos contributing to ATE, RPEt, RPEr
    }

    for i, video in enumerate(videos):
        video_path = os.path.join(base_path, video)
        print(f"Working on {i}:{video}", flush=True)
    
        R_gt, t_gt, number_of_frames = get_gt(video_path)
        preds = get_preds(video_path, model, experiment_name, number_of_frames)

        ATE, RPEt, RPEr = evaluate_pose(preds, R_gt, t_gt, normalize, video, res_save_path)

        print(ATE)

        if ATE is not None:
            errs['RPErs'] += RPEr
            errs['ATEs'] += ATE
            errs['RPEts'] += RPEt
            errs["pose_err"] += 1
            
            # Write results to txt file with fixed width columns for better alignment
            results_file.write(f"{video:<15}{ATE:<12.6f}{RPEt:<12.6f}{RPEr:<12.6f}\n")
        else:
            results_file.write(f"{video:<15}{'N/A':<12}{'N/A':<12}{'N/A':<12}\n")

    results_file.close()

    final_result = {
        "ATE": errs["ATEs"] / (errs["pose_err"]+1e-16),
        "RPEt": errs["RPEts"] / (errs["pose_err"]+1e-16),
        "RPEr": errs["RPErs"] / (errs["pose_err"]+1e-16),
        "pose_err": errs["pose_err"]
    }

    # Pretty print final_result
    print(json.dumps(final_result, indent=4))

    # Write the summary results to the end of the txt file with better formatting
    with open(f"{res_save_path}/all_results.txt", "a") as results_file:
        results_file.write("\n" + "=" * 50 + "\n")
        results_file.write("SUMMARY\n")
        results_file.write("-" * 50 + "\n")
        results_file.write(f"{'Average ATE:':<20}{final_result['ATE']:.6f}\n")
        results_file.write(f"{'Average RPEt:':<20}{final_result['RPEt']:.6f}\n")
        results_file.write(f"{'Average RPEr:':<20}{final_result['RPEr']:.6f}\n")
        results_file.write(f"{'Evaluated videos:':<20}{errs['pose_err']}\n")

    # Save the json result
    with open(f"{res_save_path}/result.json", "w") as f:
        f.write(json.dumps(final_result, indent=4))
        
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate camera pose predictions")
    parser.add_argument("--dataset", type=str, default="sintel", help="Dataset name")
    parser.add_argument("--base_path", type=str, default=None, help="Base path to dataset")
    parser.add_argument("--experiment_name", type=str, default="base", help="Experiment name")
    parser.add_argument("--model", type=str, default="dynamicBA", help="Model name")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize ground truth poses")
    
    args = parser.parse_args()
    
    eval_pose(
        dataset=args.dataset,
        base_path=args.base_path,
        experiment_name=args.experiment_name,
        model=args.model,
        normalize=args.normalize
    )


if __name__ == '__main__':
    main()