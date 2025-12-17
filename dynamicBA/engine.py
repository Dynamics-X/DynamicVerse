import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import cv2
import itertools
# from raft import load_RAFT  # Commented out original RAFT import
from unimatch.unimatch.unimatch import UniMatch  # Add UniMatch import
import logging
import json

import imageio
import matplotlib.pyplot as plt
from importlib import reload

from torchvision import transforms
import torch.nn.functional as F

from pytorch3d.ops import knn_points

from util import clean_depth_outliers, EarlyStopper, KeyFrameBuffer, l1_loss_with_uncertainty, flow_norm

from variables import CameraPoseDeltaCollection, ControlPoints, CameraIntrinsics, ControlPointsDynamic

import os
import wandb

import random

from tqdm import tqdm
import time
import glob
import sys
# Set PIL and imageio log levels to avoid "Operation on closed image" Error
import PIL.Image
PIL.Image.DEBUG = 0
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('imageio').setLevel(logging.WARNING)
logging.getLogger('imageio_ffmpeg').setLevel(logging.WARNING)

# from vis.vis_cotracker import vis

class Engine():

    def __init__(self, opt) -> None:
        """
        Initialize engine
        """
        # --- First set self.opt and device ---
        self.opt = opt 
        self.device = torch.device(self.opt.device)

        # Add default value for flow_model
        if not hasattr(self.opt, 'flow_model'):
            self.opt.flow_model = 'unimatch'
        
 
        
        # print(f"Engine initialization, using device: {self.device}") # Printed in setup_logging
        
        # --- Set paths (now guaranteed to be set by run.py main function) ---
        # Ensure BASE and output_dir have been set
        if not hasattr(self.opt, 'BASE') or not hasattr(self.opt, 'output_dir'):
            # This situation should not happen in theory, if it does need to check run.py main function logic
            logging.error("Engine initialization missing BASE or output_dir attribute!")
            # Set default values to prevent crash, but this indicates upstream logic has issues
            self.opt.BASE = os.path.abspath(os.path.join("./workdir", "default_video"))
            self.opt.output_dir = os.path.abspath(os.path.join(self.opt.BASE, "dynamicBA", "default_exp"))
            os.makedirs(self.opt.BASE, exist_ok=True)
            os.makedirs(self.opt.output_dir, exist_ok=True)
        
        # Logging system should already be configured by run.py, no longer handle log configuration here
        logging.info(f"Engine initialization: data BASE={self.opt.BASE}, output output_dir={self.opt.output_dir}")
        
    
    def init_vars(self) -> None:
        """
        Loads in the poses for optimization and initializes starting point using pred depth and pose
        """
        self.flow_norm_l1 = False

        if self.opt.loss_fn == "l1":
            self.loss_fn = l1_loss_with_uncertainty
        else:
            raise ValueError("Invalid loss type")

        # If poses already exists (loaded from state file), preserve it instead of recreating
        has_existing_poses = hasattr(self, 'poses') and self.poses is not None
        if not has_existing_poses:
            self.poses = CameraPoseDeltaCollection(self.num_frames)
            logging.info("Created new CameraPoseDeltaCollection instance")
        else:
            logging.info("Preserved loaded CameraPoseDeltaCollection instance")

        # Control points need to be recreated
        self.controlpoints_static = ControlPoints(number_of_points=self.num_points_static)
        self.controlpoints_dyn = ControlPointsDynamic(number_of_points=self.num_points_dyn, number_of_frames=self.num_frames, with_norm=False)
        
        if self.opt.opt_intrinsics:
            self.intrinsics = CameraIntrinsics(self.K[0,0], self.K[1,1])
            self.intrinsics.register_shape(self.shape)
            
    
    @torch.no_grad()
    def get_track_depths(self):

        """
        Instantiates the depth for each track
        """

        _, _, H, W  = self.depth.shape

        coords_normalized_static = torch.zeros_like(self.all_tracks_static)
        coords_normalized_static[:, :, 0] = 2.0 * self.all_tracks_static[:, :, 0] / (W - 1) - 1.0
        coords_normalized_static[:, :, 1] = 2.0 * self.all_tracks_static[:, :, 1] / (H - 1) - 1.0

        coords_normalized_dyn = torch.zeros_like(self.all_tracks_dyn)
        coords_normalized_dyn[:, :, 0] = 2.0 * self.all_tracks_dyn[:, :, 0] / (W - 1) - 1.0
        coords_normalized_dyn[:, :, 1] = 2.0 * self.all_tracks_dyn[:, :, 1] / (H - 1) - 1.0

        coords_normalized_static = coords_normalized_static.unsqueeze(2)
        coords_normalized_dyn = coords_normalized_dyn.unsqueeze(2)

        self.all_tracks_static_depth = F.grid_sample(self.depth, coords_normalized_static, align_corners=True).squeeze(1).squeeze(-1)     # F x N
        self.all_tracks_dyn_depth = F.grid_sample(self.depth, coords_normalized_dyn, align_corners=True).squeeze(1).squeeze(-1)           # F x N

    @torch.no_grad()
    def init_dyn_cp(self) -> None:

        """
        Initializes values for dynamic control points once we have some pose
        """

        for f in range(self.num_frames):   # initializes values for dynamic points

            R, t = self.get_poses([f])
            R = R[0]
            t = t[0]

            track_2d = self.all_tracks_dyn[f, :]
            vis_2d = self.all_vis_dyn[f, :]

            depth = self.get_depth(f)

            norm = None
            
            ts, ns = self.reproject(R, t, track_2d, vis_2d, depth, f, norm) # N x 3

            self.controlpoints_dyn.set_translation(f, ts)
      

    def get_depth(self, frame_idx):
        """
        Function to get depth slice from the optimizable self.depth Parameter.
        Ensures returned slice requires grad if self.depth does.
        """
        # self.depth is Parameter [F, 1, H, W]
        # Indexing with int returns [1, H, W]. Indexing with list/tensor returns [N, 1, H, W]
        # We need [H, W] for the function using it.
        depth_frame = self.depth[frame_idx, 0] # Get the specific frame, remove channel dim. Shape [H, W]
        assert len(depth_frame.shape) == 2
        return depth_frame # This tensor requires grad if self.depth does

    def get_poses(self, frame_idx):
        """
        Function to get pose, used when gt is used
        Returns rotation and translation matrices that maintain gradient connections.
        """
        # If a single integer is passed, wrap it in a list
        if isinstance(frame_idx, int):
            frame_indices = [frame_idx]
        else:
            frame_indices = frame_idx
            
        Rs, ts = self.poses(frame_indices)
        
        # If a single integer is passed, return only the first result
        if isinstance(frame_idx, int):
            Rs = Rs[0]
            ts = ts[0]
        
        # Ensure tensors are on the correct device
        Rs = Rs.to(self.device)
        ts = ts.to(self.device)
        
        # Check if tensors are connected to the computation graph
        if self.opt.flow_opt_pose and not Rs.requires_grad:
            logging.warning(f"Rotation matrix is not connected to the computation graph, which may prevent gradients from passing to the pose optimizer")
            if "rotation_pose" in self.active_optimizers:
                logging.error("rotation_pose is in active_optimizers, but the rotation matrix does not require gradients")

        if self.opt.flow_opt_pose and not ts.requires_grad:
            logging.warning(f"Translation vector is not connected to the computation graph, which may prevent gradients from passing to the pose optimizer")
            if "translation_pose" in self.active_optimizers:
                logging.error("translation_pose is in active_optimizers, but the translation vector does not require gradients")
        
        # Create a new version connected to the computation graph to avoid using assertions
        # Since assertions are ignored in production, use logging and Warning instead
        return Rs, ts

    def reinitialize_static(self):

        """
        Reinitialize static with all static points clouds
        """

        prev_3d_static = self.controlpoints_static.forward()

        all_static_filter = torch.logical_or(self.all_labels == 0, self.all_labels==2)
        vis_filter = torch.sum(self.all_vis, axis=0) > 2                                                                # things must be visible at least once
        all_static_filter = torch.logical_and(all_static_filter, vis_filter)

        self.all_tracks_static = self.all_tracks[:, all_static_filter, :]
        self.all_vis_static = self.all_vis[:, all_static_filter]
        self.all_tracks_static_init = self.track_init_frames[all_static_filter]
        self.track_init_frames_static = self.track_init_frames[all_static_filter]
        self.static_confidences = self.confidences[:, all_static_filter]

        self.num_points_static = self.all_tracks_static.shape[1] 

        self.controlpoints_static = ControlPoints(number_of_points=self.num_points_static).to(self.device)

        prev_idx = torch.where(self.all_labels[all_static_filter] == 0)[0]
        self.static_control_new_points = torch.where(self.all_labels[all_static_filter] == 2)[0]

        self.controlpoints_static.set_translation(prev_idx, prev_3d_static)                                            # sets translation for old points

        logging.info(f"Number of static points: {self.num_points_static}")
        logging.info(f"Number of dynamic points: {self.num_points_dyn}")

        self.init_BA()
        self.filter_static()



    def filter_isolated_points(self, points, percentile=98, k=5):
        """
        Filter out isolated points using PyTorch3D's optimized KNN implementation.
        
        Args:
            points: (N, 3) tensor of 3D points on CUDA
            percentile: percentile threshold for filtering (e.g., 99 removes top 1% most isolated points)
            k: number of nearest neighbors to consider
        
        Returns:
            filtered_points: (M, 3) tensor with isolated points removed
            mask: boolean mask indicating which points were kept
        """
        # Add batch dimension if needed
        if len(points.shape) == 2:
            points = points.unsqueeze(0)  # (1, N, 3)
        
        # Compute k nearest neighbors
        # knn_points returns (dists, idx) where dists is (B, P1, K) tensor of squared distances
        results = knn_points(
            points,                     # (B, P1, D) query points
            points,                     # (B, P2, D) reference points
            K=k+1,                     # number of neighbors (k+1 because point is its own neighbor)
            return_sorted=False        # we don't need sorted distances
        )
        
        # Remove the first distance (distance to self = 0)
        neighbor_distances = results.dists[0, :, 1:]   # (N, K)
        
        # Calculate average distance to k nearest neighbors
        avg_distances = torch.mean(neighbor_distances, dim=1)  # (N,)
        
        # Calculate threshold based on percentile
        threshold = torch.quantile(avg_distances, percentile/100)
        
        # Create mask for points to keep
        mask = avg_distances > threshold
        
        return mask

    def filter_static(self):


        points_3d_world = self.controlpoints_static()                                  # N x 3
        mask = self.filter_isolated_points(points_3d_world)

        self.all_vis_static[:, mask] = False


    def project(self, R, t, K, points_3d):

        """
        Projects 3d points into 2d points
        R -> 3 x 3
        t -> 3 x 1
        points_3d -> N x 3 -> points in world coordinate
        depth -> bool -> if True, returns depth as well
        """

        points_3d_cam = (torch.linalg.inv(R) @ (points_3d.T - t)).T
        points_3d_depth = points_3d_cam[:,2]

        points_2d_homo = (K @ points_3d_cam.T).T

        points_2d_homo = (points_2d_homo / (points_2d_homo[:,2:] + 1e-16))[:,:2]

        return points_2d_homo, points_3d_depth

    def reproject(self, R, t, track_2d, vis_2d, depth, f, norm = None):

        """
        Reprojects 2d points into 3d points
        R -> 3 x 3
        t -> 3 x 1
        track_2d -> N x 2, stored as x y values
        vis_2d -> N
        depth -> H x W
        """

        N,_ = track_2d.shape
        K = self.get_intrinsics_K([f])[0]

        points_2d_homo = torch.concatenate((track_2d, torch.ones((N, 1)).to(self.device)), axis=-1)

        ray = (torch.linalg.inv(K) @ points_2d_homo.T).T

        ray_norm = ray / ray[:,-1].unsqueeze(-1) # N x 3

        points_depth = depth[track_2d[:,1].long(), track_2d[:,0].long()]

        points_3d_src = ray_norm * (points_depth.unsqueeze(-1)) # N x 3

        points_3d_world_t = (R @ points_3d_src.T).T + t[:,0]

        if norm is not None:
            points_3d_world_n = (R @ norm.T).T
        else:
            points_3d_world_n = None

        return points_3d_world_t, points_3d_world_n # N x 3

    def reset_optimizer(self, lr=1e-3, patience=10):

        for optimizer in self.active_optimizers:
            del self.optimizers[optimizer]
            if optimizer in self.schedulers:
                del self.schedulers[optimizer]

        self.init_optimization(lr=lr, patience=patience)

    def filter_depth_edges(self, depth):

        """
        Filter depth edges using gradients
        """

        depth_image = depth.detach().cpu().numpy()

        # Compute the gradients along the x and y axes
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x-axis
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y-axis

        # Optionally, compute the gradient magnitude (Euclidean distance)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)

        threshold = np.percentile(grad_magnitude, 95)

        mask = grad_magnitude < threshold

        return mask


    def fuse_static_points(self, save_path):
        """
        Fuse depth point clouds into static and dynamic point clouds, without downsampling static point clouds
        """
        import open3d as o3d
        from find_anno import VideoAnnotationFinder
        import os
        import cv2

        # Create directory for saving segmentation results
        mask_save_dir = os.path.join(os.path.dirname(save_path), "masks")
        os.makedirs(mask_save_dir, exist_ok=True)
        print(f"Segmentation results will be saved to: {mask_save_dir}")

        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2)).to(self.device)
        coords_2d_homo = torch.concatenate((coords, torch.ones((len(coords),1)).float().to(self.device)), dim=1)  # N x 3
        
        # Create two separate image arrays, one for original reconstruction and one for masked reconstruction
        images_original = self.images.clone()  # Original image
        # Initialize point cloud storage arrays
        points = np.zeros((0, 3))  # Original point cloud
        rgbs = np.zeros((0, 3), np.uint8)  # Original point cloud colors
        dyn_points = []  # Original dynamic point cloud
        dyn_rgbs = []   # Original dynamic point cloud colors
        
        Rs, ts = self.get_poses(torch.arange(self.num_frames))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames))
        
        print(f"Starting point cloud processing...")
        for f in range(self.num_frames):
            R = Rs[f]
            t = ts[f]
            K = Ks[f]

            src_ray = (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.reproj_depths[f] # Use interpolated depth map
            edge_mask = self.filter_depth_edges(depth)
            edge_mask = edge_mask.flatten() # NumPy array
            
            # --- Modification: Use self.dyn_masks directly when getting static and dynamic masks --- 
            # self.dyn_masks is a PyTorch Tensor of shape [F, H, W], needs to be moved to CPU and converted to NumPy
            current_dyn_mask_f_np = self.dyn_masks[f].cpu().numpy().flatten()

            static_mask = (current_dyn_mask_f_np == 0) # Original DeVA static
            dyn_mask = (current_dyn_mask_f_np != 0)    # Original DeVA dynamic (all non-zero labels)
            # --- Modification End ---
            
            # Apply edge filtering (ensure consistent types, edge_mask is NumPy bool)
            static_mask = np.logical_and(static_mask, edge_mask)
            dyn_mask = np.logical_and(dyn_mask, edge_mask)

            # Compute 3D point cloud coordinates
            points_3d_src = src_ray_homo * depth.flatten().unsqueeze(-1) # depth on GPU, src_ray_homo on GPU
            points_3d_world = (R @ points_3d_src.T).T + t # R, t on GPU

            rgb = images_original[f].reshape((-1, 3)) # images_original on GPU

            points_3d_world_static = points_3d_world[torch.from_numpy(static_mask)]
            rgb_static = rgb[torch.from_numpy(static_mask)]
            points_3d_world_dyn = points_3d_world[torch.from_numpy(dyn_mask)]
            rgb_dyn = rgb[torch.from_numpy(dyn_mask)]
            
            points = np.vstack([points, points_3d_world_static.detach().cpu().numpy()])
            rgbs = np.vstack([rgbs, rgb_static.detach().cpu().numpy()])
            dyn_points.append(points_3d_world_dyn.detach().cpu().numpy())
            dyn_rgbs.append(rgb_dyn.detach().cpu().numpy())
            
            if f % 10 == 0:
                print(f"Processed {f}/{self.num_frames} frames")
                print(f"Number of static points in current frame: {len(points_3d_world_static)}")
                print(f"Number of dynamic points in current frame: {len(points_3d_world_dyn)}")

        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()

        print(f"\nPoint cloud processing completed:")
        print(f"Total static point cloud count: {len(points)}")
        print(f"Dynamic point cloud frame count: {len(dyn_points)}")
        
        # Saving original point cloud data
        output = {
            "static_points": points,
            "static_rgbs": rgbs,
            "dyn_points": np.array(dyn_points, dtype=object),
            "dyn_rgbs": np.array(dyn_rgbs, dtype=object),
            "c2w": c2w,
            "rgbs": images_original.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu()
        }

        print(f"Saving original point cloud data to: {save_path}")
        np.savez_compressed(save_path, **output)
        

    def save_results(self, save_fused_points=False, suffix="", save_masked=False):
        """
        Saves the results including:
        1. Fused 4D point cloud (if save_fused_points=True)
        2. Depth maps for each frame (as 16-bit PNG images)
        3. Camera poses for each frame
        
        Args:
            save_fused_points: Whether to save fused point clouds
            suffix: Suffix for saving files, e.g., '_flow' will generate 'fused_4d_flow.npz'
            save_masked: Whether to save point clouds covered by dynamic object masks with devamask+epimask
        """

        # --- Removed: depth_interpolate() is now called earlier in the optimization process ---
        # self.depth_interpolate()

        # --- Ensure self.reproj_depths (for saving) is up to date ---
        # If optimizing reproj_depths_param, need to sync the final result (if needed) back to self.reproj_depths
        if hasattr(self, 'reproj_depths_param') and self.reproj_depths_param is not None:
             self.reproj_depths = self.reproj_depths_param.detach().clone()
             # Also need to confirm if the shape of reproj_depths matches subsequent code expectations (e.g., [F, H, W])
             if len(self.reproj_depths.shape) == 4 and self.reproj_depths.shape[1] == 1:
                  self.reproj_depths = self.reproj_depths.squeeze(1) # Change to [F, H, W]
             logging.info("Updated self.reproj_depths from optimized reproj_depths_param for saving.")
        elif not hasattr(self, 'reproj_depths') or self.reproj_depths is None:
             # If reproj_depths does not exist yet (interpolation might have been skipped), try generating it once
             logging.warning("self.reproj_depths not found before saving, calling depth_interpolate now.")
             self.depth_interpolate(make_optimizable=False) # Generating non-optimized version

        # Create directories for depth maps and poses
        depth_dir = os.path.join(self.opt.output_dir, f"depth_maps{suffix}")
        os.makedirs(depth_dir, exist_ok=True)

        # Save camera poses
        all_poses = []
        all_Ks = []
        depth_metadata = []

        if save_fused_points:
            # Name point cloud file with suffix
            fused_points_file = f"fused_4d{suffix}.npz"
            fused_points_path = os.path.join(self.opt.output_dir, fused_points_file)
            
            # Use vis_4d option to decide how to save point clouds
            if hasattr(self.opt, 'vis_4d') and self.opt.vis_4d:
                # Saving original version and masked version of point clouds
                logging.info("Saving standard point cloud and point cloud covered with DeVA dynamic object masks (opacity 0.3)...")
                standard_path, masked_path = self.fuse_with_deva_masks(fused_points_path, alpha=0.3)
                logging.info(f"Standard point cloud saved to: {standard_path}")
                logging.info(f"Point cloud with DeVA mask saved to: {masked_path}")
            else:
                # Use original fuse_static_points function, saving only standard point cloud
                self.fuse_static_points(fused_points_path)   # fuse depth maps into global point clouds and saves
                logging.info(f"Standard point cloud saved to: {fused_points_path}")

        for f in range(self.num_frames):
            # Get data for current frame
            img = self.images[f].detach().cpu().numpy().squeeze()
            R,t = self.get_poses([f])
            R = R.detach().cpu().numpy().squeeze()
            t = t.detach().cpu().numpy().squeeze()
            K = self.get_intrinsics_K([f]).detach().cpu().numpy()[0]
            depth_interp = self.reproj_depths[f].detach().cpu().numpy()
            disp_aligned = (1 / (depth_interp + 1e-16))

            # Save depth map as 16-bit PNG
            depth_save_path = os.path.join(depth_dir, f"{str(f).zfill(4)}_geometric.png")
            
            # Convert depth to 16-bit unsigned integer
            # First normalize to 0-1 range
            depth_min = depth_interp.min()
            depth_max = depth_interp.max()
            depth_normalized = (depth_interp - depth_min) / (depth_max - depth_min)
            
            # Then convert to 16-bit (0-65535)
            depth_16bit = (depth_normalized * 65535).astype(np.uint16)
            
            # Save depth image
            cv2.imwrite(depth_save_path, depth_16bit)
            
            # Save depth metadata
            depth_metadata.append({
                'frame': f,
                'height': depth_interp.shape[0],
                'width': depth_interp.shape[1],
                'depth_min': depth_min,
                'depth_max': depth_max
            })

            # Store pose and intrinsics
            pose = np.eye(4)
            pose[:3,:3] = R
            pose[:3,3] = t.squeeze()
            all_poses.append(pose)
            all_Ks.append(K)

            # Save original output dict with suffix
            output_dict = {  # DISP is saved in opt_shape
                'disp': disp_aligned,
                'img': img,
                'R': R,
                't': t,
                'K': K
            }
            np.savez_compressed(os.path.join(self.opt.output_dir, f"{str(f).zfill(4)}{suffix}.npz"), **output_dict)

        # Save all poses and intrinsics with suffix
        poses_dict = {
            'poses': np.array(all_poses),
            'intrinsics': np.array(all_Ks)
        }
        np.savez_compressed(os.path.join(self.opt.output_dir, f"poses{suffix}.npz"), **poses_dict)
        
        # Save depth metadata
        np.save(os.path.join(depth_dir, "depth_metadata.npy"), depth_metadata)

    def statistical_outlier_removal(self, point_cloud, k=4, std_dev_multiplier=3.0):
        # point_cloud: (N, 3) tensor of points
        N = point_cloud.shape[0]
        
        # Compute k-nearest neighbors
        _, idx, _ = knn_points(point_cloud[None, ...], point_cloud[None, ...], K=k)
        idx = idx[0]  # (N, k)
        
        # Calculate mean distance to neighbors
        neighbor_points = point_cloud[idx]  # (N, k, 3)
        mean_distances = torch.mean(torch.norm(neighbor_points - point_cloud[:, None, :], dim=2), dim=1)
        
        # Calculate threshold distance
        mean = torch.mean(mean_distances)
        std_dev = torch.std(mean_distances)
        threshold = mean + std_dev_multiplier * std_dev
        
        # Filter out points with mean distance above threshold
        mask = mean_distances < threshold
        filtered_points = point_cloud[mask]
        
        return filtered_points, mask

    @torch.no_grad()
    def get_neighbors(self):


        """
        Get neighbors of each point
        """
        neighbors = 4

        self.all_neighbors = torch.zeros((self.num_points_dyn, neighbors)).long().to(self.device)

        all_idx = torch.arange(self.num_points_dyn).long().to(self.device)

        for i in range(self.num_points_dyn):

            f = self.track_init_frames_dyn[i].item()
            track = self.all_tracks_dyn[f]
            vis = self.all_vis_dyn[f]

            track_sel = torch.where(vis)[0]                 # index of selected tracks

            K = self.get_intrinsics_K([f])[0]

            control_pts_2d = track
            control_pts_2d_homo = torch.concatenate((control_pts_2d, torch.ones((control_pts_2d.shape[0],1)).float().to(control_pts_2d.device)), dim=1)  # N x 3

            src_ray =  (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.get_depth(f)
            depth_src = self.interpolate(depth.unsqueeze(-1), control_pts_2d)

            points_3d_src = (src_ray_homo * depth_src)

            query_points_3d_src = points_3d_src[i:i+1]
            tgt_points_3d_src = points_3d_src[track_sel]

            knn_result = knn_points(query_points_3d_src.unsqueeze(0), tgt_points_3d_src.unsqueeze(0),  K=neighbors+1)
            idx = knn_result.idx[0]         # (N, k)

            self.all_neighbors[i] = all_idx[track_sel][idx[0, 1:]]


    def depth_interpolate(self, make_optimizable=False):

        """
        Scale interpolation using 3d knn in unidepth reprojection
        """
        
        # --- Add log to confirm mask usage ---
        logging.info("Starting depth interpolation - confirming mask usage:")
        
        # Check if masks unaffected by EPI exist
        if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
            logging.info("Using self.dyn_masks_filters_deva (DeVA masks unaffected by EPI) for depth interpolation")
            use_deva_only_mask = True
        else:
            logging.warning("dyn_masks_filters_deva does not exist, falling back to self.dyn_masks (possibly affected by EPI)")
            logging.warning("Suggest checking mask generation process to ensure depth interpolation accuracy")
            use_deva_only_mask = False
        
        # Statistic mask label distribution (showing details for only the first few frames to avoid excessive logs)
        logging.info(f"Processing depth interpolation for all {self.num_frames} frames")
        logging.info("Detailed mask statistics for the first 3 frames shown below:")
        
        for f in range(min(3, self.num_frames)):
            if use_deva_only_mask:
                unique_labels_deva = np.unique(self.dyn_masks_filters_deva[f])
                logging.info(f"Frame {f} - labels in dyn_masks_filters_deva: {unique_labels_deva}")
                
                # Count pixels for each label
                static_pixels = np.sum(self.dyn_masks_filters_deva[f] == 0)
                dyn_pixels = np.sum(self.dyn_masks_filters_deva[f] == 1)
                total_pixels = self.dyn_masks_filters_deva[f].size
                logging.info(f"Frame {f} - Static region: {static_pixels} ({static_pixels/total_pixels*100:.1f}%), DeVA dynamic region: {dyn_pixels} ({dyn_pixels/total_pixels*100:.1f}%)")
                
                # Compare with masks possibly affected by EPI
                if hasattr(self, 'dyn_masks_filters'):
                    epi_pixels = np.sum(self.dyn_masks_filters[f] == 3)
                    if epi_pixels > 0:
                        logging.info(f"Frame {f} - Comparison: dyn_masks_filters has {epi_pixels} EPI dynamic pixels, but depth interpolation will ignore them")
            else:
                unique_labels = torch.unique(self.dyn_masks[f])
                logging.info(f"Frame {f} - labels in dyn_masks: {unique_labels.cpu().numpy()}")
        
        if use_deva_only_mask:
            logging.info("Depth interpolation will use pure DeVA masks unaffected by EPI to ensure geometric consistency")
        
        # Show overall statistics
        if use_deva_only_mask and self.num_frames > 3:
            total_static_pixels = np.sum(self.dyn_masks_filters_deva == 0)
            total_dyn_pixels = np.sum(self.dyn_masks_filters_deva == 1) 
            total_border_pixels = np.sum(self.dyn_masks_filters_deva == 2)
            total_all_pixels = self.dyn_masks_filters_deva.size
            logging.info(f"All frames statistics - Static: {total_static_pixels} ({total_static_pixels/total_all_pixels*100:.1f}%), DeVA dynamic: {total_dyn_pixels} ({total_dyn_pixels/total_all_pixels*100:.1f}%), Border: {total_border_pixels} ({total_border_pixels/total_all_pixels*100:.1f}%)")

        self.reproj_depths = []

        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2)).to(self.device)
        coords_2d_homo = torch.concatenate((coords, torch.ones((len(coords),1)).float().to(self.device)), dim=1)  # N x 3
        

        scene_points = self.controlpoints_static()
        self.scene_scale = torch.mean(torch.quantile(scene_points, 0.9, dim=0) - torch.quantile(scene_points, 0.1, dim=0))

        logging.info(f"Starting depth interpolation processing for all {self.num_frames} frames...")
        processed_static_regions = 0
        processed_dynamic_regions = 0

        for f in range(self.num_frames):

            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]

            K = self.get_intrinsics_K([f])[0]
            src_ray =  (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.get_depth(f)

            depthmap_res = torch.zeros_like(depth)

            if use_deva_only_mask:
                # Use DeVA masks unaffected by EPI
                deva_mask_f = self.dyn_masks_filters_deva[f]
                
                # Processing static region (label 0) - non-DeVA region
                static_mask = (deva_mask_f == 0)
                if np.any(static_mask):
                    self.fuse_helper_deva(R, t, depth, 0, depthmap_res, f, coords, src_ray_homo, static_mask)
                    processed_static_regions += 1
                
                # Processing dynamic region (label 1 = DeVA dynamic)
                dyn_mask = (deva_mask_f == 1)
                if np.any(dyn_mask):
                    self.fuse_helper_deva(R, t, depth, 1, depthmap_res, f, coords, src_ray_homo, dyn_mask)
                    processed_dynamic_regions += 1
                
                # Other regions (label 2 = border region) keep original depth
                other_mask = (deva_mask_f == 2)
                if np.any(other_mask):
                    depthmap_res[other_mask] = depth[other_mask]
                    
            else:
                # Fallback to original method
                labels = torch.unique(self.dyn_masks[f])    # get unique labels for this frame
                for label in labels:
                    self.fuse_helper(R, t, depth, label, depthmap_res, f, coords, src_ray_homo)

            self.reproj_depths.append(depthmap_res.detach())   # add res depthmap
            
            # Show progress periodically
            if (f + 1) % 10 == 0 or f == self.num_frames - 1:
                logging.info(f"Depth interpolation progress: {f + 1}/{self.num_frames} frames completed ({(f + 1)/self.num_frames*100:.1f}%)")

        self.reproj_depths = torch.stack(self.reproj_depths)
        
        # Add completion confirmation and statistics
        logging.info(f"✅ Depth interpolation completed! Successfully processed all {self.num_frames} frames")
        if use_deva_only_mask:
            logging.info(f"Processing stats: Static region interpolation {processed_static_regions} times, DeVA dynamic region interpolation {processed_dynamic_regions} times")
            logging.info(f"Used pure DeVA masks unaffected by EPI, ensuring geometric consistency")
        else:
            logging.info(f"Used original DeVA instance masks for depth interpolation")

        # If optimizable parameters need to be created
        if make_optimizable:
            self.reproj_depths_param = torch.nn.Parameter(self.reproj_depths.clone(), requires_grad=True)
            logging.info("Created optimizable depth parameter (reproj_depths_param)")

        assert(torch.sum(torch.isnan(self.reproj_depths))==0)
        if torch.sum(self.reproj_depths==0)> 0:
            breakpoint()


    def fuse_with_deva_masks(self, save_path, alpha=0.3):
        """
        Fuse depth point clouds while creating a version with DeVA masks
        Saving two sets of results: 1. Standard static + dynamic point clouds, 2. Point clouds with DeVA masks (each dynamic instance marked with a different color)
        
        Args:
            save_path: Saving path
            alpha: Dynamic region transparency, default 0.3
            
        Returns:
            tuple: (standard_path, masked_path)
        """
        import os
        import cv2
        import numpy as np

        # Create saving directory
        mask_save_dir = os.path.join(os.path.dirname(save_path), "masks")
        os.makedirs(mask_save_dir, exist_ok=True)
        logging.info(f"Mask visualization will be saved to: {mask_save_dir}")

        # Ensure coordinates are on the correct device
        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2), device=self.device)
        coords_2d_homo = torch.cat((coords, torch.ones((len(coords),1), device=self.device)), dim=1)  # N x 3
        
        # Create copy of original images
        images_original = self.images.clone().to(self.device)  # Ensure original images are on the correct device
        
        # Initialize point cloud storage arrays
        # 1. Standard point cloud
        static_points = np.zeros((0, 3))  # Static point cloud
        static_rgbs = np.zeros((0, 3), np.uint8)  # Static point cloud colors
        dyn_points = []  # Dynamic point cloud list (by frame)
        dyn_rgbs = []   # Dynamic point cloud color list (by frame)
        
        # 2. Masked point cloud
        masked_static_points = np.zeros((0, 3))  # Masked static point cloud
        masked_static_rgbs = np.zeros((0, 3), np.uint8)  # Masked static point cloud colors
        masked_dyn_points = []  # Masked dynamic point cloud list (by frame)
        masked_dyn_rgbs = []   # Masked dynamic point cloud color list (by frame)
        
        # Get camera parameters
        Rs, ts = self.get_poses(torch.arange(self.num_frames, device=self.device))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames, device=self.device))
        
        # Check if dyn_masks_filters_deva exists
        if not hasattr(self, 'dyn_masks_filters_deva'):
            logging.warning("dyn_masks_filters_deva does not exist, attempting to use dyn_masks_filters")
            if hasattr(self, 'dyn_masks_filters'):
                logging.info("Found dyn_masks_filters, using it as alternative")
                self.dyn_masks_filters_deva = self.dyn_masks_filters
            else:
                logging.warning("dyn_masks_filters also does not exist, using dyn_masks")
                # Create simple binary mask
                h, w = self.shape
                num_frames = self.dyn_masks.shape[0]
                self.dyn_masks_filters_deva = np.zeros((num_frames, h, w), dtype=np.uint8)
                for f in range(num_frames):
                    mask = self.dyn_masks[f].cpu().numpy()
                    self.dyn_masks_filters_deva[f] = (mask > 0).astype(np.uint8)
        
        # Find all instance IDs (assuming dyn_masks contains instance IDs)
        instance_ids = set()
        if hasattr(self, 'dyn_masks'):
            # Use original dyn_masks to get all instance IDs
            for f in range(self.num_frames):
                unique_ids = torch.unique(self.dyn_masks[f]).cpu().numpy()
                for id in unique_ids:
                    if id > 0:  # Exclude background (ID=0)
                        instance_ids.add(int(id))
        
        if not instance_ids:
            # If instance IDs cannot be found, use default values
            instance_ids = {1}  # Default to only one instance (ID=1)
            logging.warning(f"No valid instance IDs found, using default ID: {instance_ids}")
        else:
            logging.info(f"Found the following instance IDs: {instance_ids}")
        
        # Assign a unique color to each instance ID
        instance_colors = {}
        
        # Use fixed color scheme (RGB format, matching final expected display color)
        fixed_colors = [
            [255, 0, 0],    # Instance 1: Red
            [0, 255, 0],    # Instance 2: Green
            [0, 0, 255]     # Instance 3: Blue
        ]
        
        # Assign colors to instances
        import random
        for instance_id in sorted(instance_ids):
            if instance_id <= 3:
                # First three instances use fixed colors
                color_idx = instance_id - 1
                instance_colors[instance_id] = np.array(fixed_colors[color_idx])
            else:
                # Generate random colors for other instances (RGB format)
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                instance_colors[instance_id] = np.array([r, g, b])
        
        logging.info(f"Colors assigned for {len(instance_colors)} instances")
        for instance_id, color in instance_colors.items():
            color_name = ""
            if np.array_equal(color, [255, 0, 0]):
                color_name = "Red"
            elif np.array_equal(color, [0, 255, 0]):
                color_name = "Green"
            elif np.array_equal(color, [0, 0, 255]):
                color_name = "Blue"
            else:
                color_name = f"RGB({color[0]},{color[1]},{color[2]})"
            logging.info(f"  Instance {instance_id}: {color_name} {color}")
        
        logging.info(f"Starting point cloud and DeVA mask processing...")
        for f in range(self.num_frames):
            # Ensure all tensors are on the same device
            R = Rs[f].to(self.device)
            t = ts[f].to(self.device)
            K = Ks[f].to(self.device)

            # Use torch.matmul instead of @ operator for clearer matrix multiplication
            src_ray = torch.matmul(torch.linalg.inv(K), coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            # Get depth map
            depth = self.reproj_depths[f].to(self.device)  # Ensure depth is on the correct device
            # Apply edge filtering
            edge_mask = self.filter_depth_edges(depth)
            edge_mask = edge_mask.flatten()  # NumPy array
            
            # --- Get DeVA mask ---
            # Use dyn_masks_filters_deva to get DeVA mask
            # Label 0: Static region
            # Label 1: DeVA dynamic region
            current_deva_mask = self.dyn_masks_filters_deva[f].flatten()
            static_mask = (current_deva_mask == 0)  # Static region
            dyn_mask = (current_deva_mask == 1)     # DeVA dynamic region
            
            # Apply edge filtering
            static_mask = np.logical_and(static_mask, edge_mask)
            dyn_mask = np.logical_and(dyn_mask, edge_mask)
            
            # If processing multiple instances, use dyn_masks to get instance IDs
            instance_masks = {}
            if len(instance_ids) > 1 and hasattr(self, 'dyn_masks'):
                current_dyn_mask_f_np = self.dyn_masks[f].cpu().numpy().flatten()
                for instance_id in instance_ids:
                    # Create mask for each instance ID
                    instance_mask = (current_dyn_mask_f_np == instance_id)
                    # Ensure instance is within DeVA dynamic region and passes edge filtering
                    instance_mask = np.logical_and(instance_mask, dyn_mask)
                    if np.any(instance_mask):
                        instance_masks[instance_id] = instance_mask
            else:
                # If only one instance or no instance info, treat all dynamic regions as one instance
                if 1 in instance_ids:
                    instance_masks[1] = dyn_mask
                else:
                    instance_id = next(iter(instance_ids))
                    instance_masks[instance_id] = dyn_mask
            
            # Saving mask visualization (for representative frames only)
            if f % 10 == 0:  # Save mask visualization every 10 frames
                # Create mask image
                mask_img = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
                # Static region - Green
                mask_img[static_mask.reshape(self.shape)] = [0, 255, 0]
                # DeVA dynamic region - Red
                mask_img[dyn_mask.reshape(self.shape)] = [0, 0, 255]
                
                # Saving mask image
                cv2.imwrite(os.path.join(mask_save_dir, f"frame_{f:04d}_deva_mask.png"), mask_img)
            
            # Compute 3D point cloud coordinates - Ensure all computations are on the same device
            points_3d_src = src_ray_homo * depth.flatten().unsqueeze(-1)
            points_3d_world = torch.matmul(R, points_3d_src.T).T + t.squeeze(-1)
            
            # Get RGB colors - Ensure they are on the same device
            rgb = images_original[f].reshape((-1, 3))
            
            # Convert mask to pytorch tensor and move to correct device
            static_mask_tensor = torch.from_numpy(static_mask).to(self.device)
            dyn_mask_tensor = torch.from_numpy(dyn_mask).to(self.device)
            
            # === Processing Static Region ===
            if static_mask_tensor.sum() > 0:
                points_3d_world_static = points_3d_world[static_mask_tensor]
                rgb_static = rgb[static_mask_tensor]
                
                # Add to standard static point cloud
                static_points = np.vstack([static_points, points_3d_world_static.detach().cpu().numpy()])
                static_rgbs = np.vstack([static_rgbs, rgb_static.detach().cpu().numpy()])
                
                # Add to masked static point cloud
                masked_static_points = np.vstack([masked_static_points, points_3d_world_static.detach().cpu().numpy()])
                masked_static_rgbs = np.vstack([masked_static_rgbs, rgb_static.detach().cpu().numpy()])
            
            # === Processing Dynamic Region ===
            # Initialize current frame's dynamic point cloud
            frame_dyn_points = []
            frame_dyn_rgbs = []
            frame_masked_dyn_points = []
            frame_masked_dyn_rgbs = []
            
            # If processing multiple instances, use dyn_masks to get instance IDs
            if len(instance_ids) > 1 and hasattr(self, 'dyn_masks'):
                current_dyn_mask_f_np = self.dyn_masks[f].cpu().numpy().flatten()
                for instance_id in instance_ids:
                    # Create mask for each instance ID
                    instance_mask = (current_dyn_mask_f_np == instance_id)
                    # Ensure instance is within DeVA dynamic region and passes edge filtering
                    instance_mask = np.logical_and(instance_mask, dyn_mask)
                    
                    if np.any(instance_mask):
                        # Convert instance mask to tensor and move to correct device
                        instance_mask_tensor = torch.from_numpy(instance_mask).to(self.device)
                        
                        # Get point cloud for current instance
                        instance_points = points_3d_world[instance_mask_tensor]
                        instance_rgbs = rgb[instance_mask_tensor]
                        
                        # Standard dynamic point cloud (keep original colors)
                        frame_dyn_points.extend(instance_points.detach().cpu().numpy())
                        frame_dyn_rgbs.extend(instance_rgbs.detach().cpu().numpy())
                        
                        # Masked dynamic point cloud (apply color blending)
                        instance_color = instance_colors.get(instance_id, np.array([0, 0, 255]))  # Default Blue
                        instance_rgbs_np = instance_rgbs.detach().cpu().numpy()
                        blended_colors = instance_rgbs_np * (1-alpha) + instance_color * alpha
                        
                        frame_masked_dyn_points.extend(instance_points.detach().cpu().numpy())
                        frame_masked_dyn_rgbs.extend(blended_colors.astype(np.uint8))
            else:
                # If only one instance or no instance info, treat all dynamic regions as one instance
                if dyn_mask_tensor.sum() > 0:
                    points_3d_world_dyn = points_3d_world[dyn_mask_tensor]
                    rgb_dyn = rgb[dyn_mask_tensor]
                    
                    # Standard dynamic point cloud (keep original colors)
                    frame_dyn_points.extend(points_3d_world_dyn.detach().cpu().numpy())
                    frame_dyn_rgbs.extend(rgb_dyn.detach().cpu().numpy())
                    
                    # Masked dynamic point cloud (apply color blending)
                    instance_id = 1 if 1 in instance_ids else next(iter(instance_ids))
                    instance_color = instance_colors.get(instance_id, np.array([0, 0, 255]))  # Default Blue
                    rgb_dyn_np = rgb_dyn.detach().cpu().numpy()
                    blended_colors = rgb_dyn_np * (1-alpha) + instance_color * alpha
                    
                    frame_masked_dyn_points.extend(points_3d_world_dyn.detach().cpu().numpy())
                    frame_masked_dyn_rgbs.extend(blended_colors.astype(np.uint8))
            
            # Add current frame's dynamic point cloud to list
            if frame_dyn_points:
                dyn_points.append(np.array(frame_dyn_points))
                dyn_rgbs.append(np.array(frame_dyn_rgbs))
            else:
                dyn_points.append(np.zeros((0, 3)))
                dyn_rgbs.append(np.zeros((0, 3), dtype=np.uint8))
                
            if frame_masked_dyn_points:
                masked_dyn_points.append(np.array(frame_masked_dyn_points))
                masked_dyn_rgbs.append(np.array(frame_masked_dyn_rgbs))
            else:
                masked_dyn_points.append(np.zeros((0, 3)))
                masked_dyn_rgbs.append(np.zeros((0, 3), dtype=np.uint8))
            
            if f % 10 == 0:
                logging.info(f"Processed {f}/{self.num_frames} frames")
                logging.info(f"  Static point count: {static_mask_tensor.sum().item()}")
                logging.info(f"  Dynamic point count: {len(frame_dyn_points)}")
        
        # Prepare camera poses
        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()
        
        logging.info(f"\nPoint cloud processing completed:")
        logging.info(f"Total standard static point clouds: {len(static_points)}")
        logging.info(f"Standard dynamic point cloud frames: {len(dyn_points)}")
        logging.info(f"Total masked static point clouds: {len(masked_static_points)}")
        logging.info(f"Masked dynamic point cloud frames: {len(masked_dyn_points)}")
        
        # Saving standard point cloud data
        standard_output = {
            "static_points": static_points,
            "static_rgbs": static_rgbs,
            "dyn_points": np.array(dyn_points, dtype=object),
            "dyn_rgbs": np.array(dyn_rgbs, dtype=object),
            "c2w": c2w,
            "rgbs": images_original.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu()
        }
        
        # Saving masked point cloud data
        masked_output = {
            "static_points": masked_static_points,  # Background static points
            "static_rgbs": masked_static_rgbs,      # Background static point colors
            "dyn_points": np.array(masked_dyn_points, dtype=object),  # Dynamic instance points (by frame)
            "dyn_rgbs": np.array(masked_dyn_rgbs, dtype=object),      # Dynamic instance point colors (by frame)
            "c2w": c2w,
            "rgbs": images_original.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu(),
            "instance_colors": instance_colors  # Saving instance color mapping for future use
        }
        
        # Saving two versions of point clouds
        standard_path = save_path
        masked_path = os.path.join(os.path.dirname(save_path), "deva_masked_" + os.path.basename(save_path))
        
        logging.info(f"Saving standard point cloud data to: {standard_path}")
        np.savez_compressed(standard_path, **standard_output)
        
        logging.info(f"Saving point cloud data with DeVA mask to: {masked_path}")
        np.savez_compressed(masked_path, **masked_output)
        
        return standard_path, masked_path


    def fuse_helper(self, R, t, depth, label, depthmap_res, f, coords, src_ray_homo, log_this_call=False):

        """
        Fuses results using label
        """

        if label == 0:

            vis = self.all_vis_static[f]             # N
            track_sel = torch.where(vis)[0]
            control_pts_3D_world = self.controlpoints_static(track_sel)        # N x 3  -->  static points

        else:

            track_dyn = self.all_tracks_dyn[f]
            vis = self.all_vis_dyn[f]
            track_dyn = track_dyn[vis]
            track_sel_dyn = torch.where(vis)[0]
            control_pts, _  = self.controlpoints_dyn(frames=f, points=track_sel_dyn)    # N x 1 x 3
            control_pts_3D_world = control_pts.squeeze(1)                                    # N x 3

        mask = self.dyn_masks[f] == label

        if len(control_pts_3D_world) == 0:
            depthmap_res[mask] = depth[mask]     # no alignment if we have no control points
            return
    
        K = self.get_intrinsics_K([f])[0]
        control_pts_2D, control_points_depth = self.project(R, t, K, control_pts_3D_world)
        
        in_range = torch.logical_and(torch.logical_and(control_pts_2D[:,0] >= 0, control_pts_2D[:,0] < self.shape[1]), torch.logical_and(control_pts_2D[:,1] >= 0, control_pts_2D[:,1] < self.shape[0]))   # filter in-view after projection
        in_range = torch.logical_and(in_range, control_points_depth>0)

        control_pts_2D = control_pts_2D[in_range]
        control_points_depth = control_points_depth[in_range]
        unidepth_vals = self.interpolate(depth.unsqueeze(-1), control_pts_2D).squeeze(-1)

        if torch.sum(in_range) == 0:
            depthmap_res[mask] = depth[mask]     # no alignment if we have no control points
            return
        
        depth_src = depth[coords[:,1].long(), coords[:,0].long()]

        points_3d_src = (src_ray_homo * depth_src.unsqueeze(-1))   # N x 3
        points_3d_world = (R @ points_3d_src.T).T + t[:,0]

        points_3d_world_grid = points_3d_world.reshape((self.shape[0], self.shape[1], 3))
        control_pts_3D_world_unidepth = points_3d_world_grid[control_pts_2D[:, 1].long(), control_pts_2D[:,0].long()]  # axis seems to be flipped

        points_3d_query = points_3d_world[mask.flatten()]

        knn_result = knn_points(points_3d_query.unsqueeze(0), control_pts_3D_world_unidepth.unsqueeze(0),  K=3)   # we find nn using reproject unidepth 3D instead

        idx = knn_result.idx[0]  # Shape: (num_points1, k)
        dists = knn_result.dists[0]

        weights = 1 / (dists + 1e-8) 

        weights_sum = weights.sum(dim=-1, keepdim=True)
        probabilities = weights / (weights_sum)
        scale = control_points_depth / unidepth_vals

        scales = torch.sum(scale[idx] * probabilities, axis=1)
    
        depthmap_res[mask] = depth[mask] * scales

        if torch.sum(scales < 0) > 0:
            breakpoint()

        for f in range(self.num_frames):

            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]

            K = self.get_intrinsics_K([f])[0]
            src_ray =  (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.get_depth(f)

            depthmap_res = torch.zeros_like(depth)

            if use_deva_only_mask:
                # Use DeVA masks unaffected by EPI
                deva_mask_f = self.dyn_masks_filters_deva[f]
                
                # Processing static region (label 0) - non-DeVA region
                static_mask = (deva_mask_f == 0)
                if np.any(static_mask):
                    self.fuse_helper_deva(R, t, depth, 0, depthmap_res, f, coords, src_ray_homo, static_mask)
                    processed_static_regions += 1
                
                # Processing dynamic region (label 1 = DeVA dynamic)
                dyn_mask = (deva_mask_f == 1)
                if np.any(dyn_mask):
                    self.fuse_helper_deva(R, t, depth, 1, depthmap_res, f, coords, src_ray_homo, dyn_mask)
                    processed_dynamic_regions += 1
                
                # Other regions (label 2 = border region) keep original depth
                other_mask = (deva_mask_f == 2)
                if np.any(other_mask):
                    depthmap_res[other_mask] = depth[other_mask]
                    
            else:
                # Fallback to original method
                labels = torch.unique(self.dyn_masks[f])    # get unique labels for this frame
                for label in labels:
                    self.fuse_helper(R, t, depth, label, depthmap_res, f, coords, src_ray_homo)

            self.reproj_depths.append(depthmap_res.detach())   # add res depthmap
            
            # Show progress periodically
            if (f + 1) % 10 == 0 or f == self.num_frames - 1:
                logging.info(f"Depth interpolation progress: {f + 1}/{self.num_frames} frames completed ({(f + 1)/self.num_frames*100:.1f}%)")

        self.reproj_depths = torch.stack(self.reproj_depths)
        
        # Add completion confirmation and statistics
        logging.info(f"✅ Depth interpolation completed! Successfully processed all {self.num_frames} frames")
        if use_deva_only_mask:
            logging.info(f"Processing stats: Static region interpolation {processed_static_regions} times, DeVA dynamic region interpolation {processed_dynamic_regions} times")
            logging.info(f"Used pure DeVA masks unaffected by EPI, ensuring geometric consistency")
        else:
            logging.info(f"Used original DeVA instance masks for depth interpolation")

        # If optimizable parameters need to be created
        if make_optimizable:
            self.reproj_depths_param = torch.nn.Parameter(self.reproj_depths.clone(), requires_grad=True)
            logging.info("Created optimizable depth parameter (reproj_depths_param)")

        assert(torch.sum(torch.isnan(self.reproj_depths))==0)
        if torch.sum(self.reproj_depths==0)> 0:
            breakpoint()

        # --- Process DeVA masks for consistent instance IDs ---
        if not all_masks_deva:
             logging.error("No DeVA masks were loaded.")
             h, w = self.shape
             num_f = self.num_frames
             self.dyn_masks = torch.zeros((num_f, h, w), dtype=torch.int64)
             self.dyn_masks_filters = (torch.zeros((num_f, h, w), dtype=torch.uint8) + 2).numpy() # All rejected
             logging.warning("No DeVA masks loaded, cannot generate filters.")
             return

        masks_deva_np = np.array(all_masks_deva)
        if masks_deva_np.size == 0: # Check if the array is empty after creation
            logging.error("Final DeVA numpy array is empty after attempting to load masks.")
        else:
            # Only process unique IDs if masks_deva_np_array is not empty
            _, unique_deva_ids_np = np.unique(masks_deva_np, return_inverse=True)
            self.dyn_masks = torch.tensor(unique_deva_ids_np.reshape(masks_deva_np.shape), dtype=torch.int64, device='cpu')

        # --- Generate the final filtered mask (self.dyn_masks_filters) ---
        dyn_masks_filters_list = []
        # New: Add a separate list to store only DeVA masks without EPI influence
        dyn_masks_filters_deva_list = []
        logging.info("Generating final filtered masks (Static=0, DeVA Dyn=1, Rejected=2, EPI Dyn=3 - EPI has highest priority)...")
        
        num_loaded_frames = masks_deva_np.shape[0] 
        
        for f in range(num_loaded_frames):
            mask_dyn = masks_deva_np[f] 
            epi_mask = all_masks_epi[f]  

            # Initialize filter mask: 2 = rejected/boundary
            mask_dyn_pro = np.full(mask_dyn.shape, 2, dtype=np.uint8) 

            # Erode static region (DeVA background ID 0)
            static_mask_eroded = cv2.erode((mask_dyn == 0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=3).astype(np.bool_)
            mask_dyn_pro[static_mask_eroded] = 0 # Label core static first

            # Erode dynamic region (DeVA non-background IDs)
            dyn_mask_eroded = cv2.erode((mask_dyn != 0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=2).astype(np.bool_)
            
            # Label core DeVA dynamic ONLY if it wasn't already marked as core static
            deva_dyn_final_mask = dyn_mask_eroded & (mask_dyn_pro != 0)
            mask_dyn_pro[deva_dyn_final_mask] = 1 
            
            # Create a copy of the mask at this point before EPI is applied
            # This preserves all original DeVA dynamic regions (label=1)
            mask_dyn_deva_only = mask_dyn_pro.copy()
            dyn_masks_filters_deva_list.append(mask_dyn_deva_only)
            
            # Apply EPI dynamic mask, overwriting any previous label
            if epi_mask is not None and epi_mask.shape == mask_dyn_pro.shape:
                epi_pixels = np.sum(epi_mask)
                mask_dyn_pro[epi_mask] = 3 # EPI dynamic overwrites DeVA dynamic and static
                logging.info(f"Frame {f}: Applied EPI mask, {epi_pixels} pixels marked as EPI dynamic (label 3)")
                
                # Count pixels that were changed from DeVA dynamic (1) to EPI dynamic (3)
                deva_to_epi = np.sum((mask_dyn_deva_only == 1) & epi_mask)
                if deva_to_epi > 0:
                    logging.warning(f"Frame {f}: {deva_to_epi} pixels changed from DeVA dynamic to EPI dynamic")
                
                # Statistics for different label types
                unique_labels, counts = np.unique(mask_dyn_pro, return_counts=True)
                label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
                logging.debug(f"Frame {f} label statistics: {label_stats}")
            else:
                if epi_mask is not None:
                    logging.warning(f"Frame {f}: EPI mask exists but shape mismatch, cannot apply. EPI shape: {epi_mask.shape}, required shape: {mask_dyn_pro.shape}")
                # No EPI mask for this frame or shape mismatch, log existing labels
                unique_labels, counts = np.unique(mask_dyn_pro, return_counts=True)
                label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
                logging.debug(f"Frame {f} label statistics (no EPI): {label_stats}")

            dyn_masks_filters_list.append(mask_dyn_pro)

        # Assign the final combined result to self.dyn_masks_filters
        self.dyn_masks_filters = np.array(dyn_masks_filters_list).astype(np.uint8)
        
        # Assign the DeVA-only masks (without EPI influence) to a new attribute
        self.dyn_masks_filters_deva = np.array(dyn_masks_filters_deva_list).astype(np.uint8)
        logging.info("Created separate DeVA-only mask (dyn_masks_filters_deva) for dynamic optimization")
        
        # Log statistics comparing the two mask types
        total_deva_dynamic_original = np.sum(self.dyn_masks_filters_deva == 1)
        total_deva_dynamic_after_epi = np.sum(self.dyn_masks_filters == 1)
        total_epi_dynamic = np.sum(self.dyn_masks_filters == 3)
        
        logging.info(f"Mask statistics: Original DeVA dynamic pixels: {total_deva_dynamic_original}")
        logging.info(f"Mask statistics: DeVA dynamic pixels after EPI applied: {total_deva_dynamic_after_epi}")
        logging.info(f"Mask statistics: EPI dynamic pixels: {total_epi_dynamic}")
        logging.info(f"Mask statistics: Pixels changed from DeVA to EPI: {total_deva_dynamic_original - total_deva_dynamic_after_epi}")
        
        # Remove other older filter attributes if they exist
        if hasattr(self, 'dyn_masks_filters_epi'): delattr(self, 'dyn_masks_filters_epi')
        if hasattr(self, 'dyn_masks_filters_combined'): delattr(self, 'dyn_masks_filters_combined')

        logging.info("Finished generating final filtered masks (self.dyn_masks_filters and self.dyn_masks_filters_deva).")

        # Now based on dyn_masks_filters create is_static_strict and is_static_strict_tensor
        h, w = self.shape
        num_f_effective = self.dyn_masks_filters.shape[0] # Use actual frame count
        self.is_static_strict = np.zeros((num_f_effective, h, w), dtype=bool)
        for f_idx in range(num_f_effective):
            # Modification: Only label 0 is strictly static
            # Label 1=DeVA Dynamic, Label 2=Border/Rejected, Label 3=EPI Dynamic
            self.is_static_strict[f_idx] = (self.dyn_masks_filters[f_idx] == 0)
            
            # Count pixels for each label, for debugging
            label_counts = np.bincount(self.dyn_masks_filters[f_idx].flatten(), minlength=4)
            logging.debug(f"Frame {f_idx} Label stats: Static(0)={label_counts[0]}, DeVA Dyn(1)={label_counts[1]}, Border/Rejected(2)={label_counts[2]}, EPI Dyn(3)={label_counts[3]}")

        if hasattr(self, 'is_static_strict') and self.is_static_strict.size > 0:
            self.is_static_strict_tensor = torch.from_numpy(self.is_static_strict).bool().to(self.device)
            logging.info(f"Created/Updated is_static_strict_tensor based on dyn_masks_filters == 0, shape: {self.is_static_strict_tensor.shape}")
            
            # Add more detailed log output, showing static region ratio
            total_pixels = self.is_static_strict.size
            static_pixels = np.sum(self.is_static_strict)
            logging.info(f"Strict static region ratio: {static_pixels}/{total_pixels} ({static_pixels/total_pixels*100:.2f}%)")
        elif hasattr(self, 'is_static_strict_tensor'): 
            del self.is_static_strict_tensor
        
        logging.info("Finished generating masks and is_static_strict_tensor.")

    def float_to_heat_rgb(self, value):
        """
        Convert a float in the range 0-1 to an RGB color based on the heat colormap.
        
        Parameters:
        value (float): A float value between 0 and 1.
        
        Returns:
        tuple: A tuple representing the RGB color.
        """
        if not 0 <= value <= 1:
            raise ValueError("Value must be within the range [0, 1]")
        
        # Get the colormap
        colormap = plt.get_cmap('cool')
        
        # Get the color corresponding to the value
        rgba = colormap(value)
        
        # Convert the RGBA color to RGB
        rgb = tuple(int(255 * c) for c in rgba[:3])
        
        return rgb

    def load_depths(self):

        """
        Get depth map and intrinsics from unidepth.
        Make depth map an optimizable parameter.
        """
        self.depth_raw = [] # Store raw numpy array if needed elsewhere

        depth_save_base = os.path.join(self.opt.workdir, self.opt.video_name, self.opt.depth_dir)
        depth_save_path_depth =  os.path.join(depth_save_base, "depth.npy")

        # Load depth numpy array
        loaded_depth_full = np.load(depth_save_path_depth)
        
        # 【Core Modification】: Truncate depth map based on self.num_frames
        loaded_depth = loaded_depth_full[:self.num_frames]
        
        self.depth_raw = loaded_depth # Keep raw numpy if needed
        logging.info(f"Loaded depth numpy array and truncated to shape: {loaded_depth.shape}")

        depth_tensor = torch.tensor(loaded_depth).float().to(self.device)
        # depth_save_path_depth =  os.path.join(depth_save_base, "depth.npy")

        # # Load depth numpy array
        # loaded_depth = np.load(depth_save_path_depth)
        # self.depth_raw = loaded_depth # Keep raw numpy if needed
        # logging.info(f"Loaded depth numpy array with shape: {loaded_depth.shape}")

        # depth_tensor = torch.tensor(loaded_depth).float().to(self.device)

        # Ensure depth tensor has shape [F, 1, H, W] before wrapping in Parameter
        if len(depth_tensor.shape) == 3: # Assume input is [F, H, W]
            depth_tensor = depth_tensor.unsqueeze(1) # Add channel dim -> [F, 1, H, W]
            logging.info("Reshaped 3D depth to 4D by adding channel dim.")
        elif len(depth_tensor.shape) == 4: # Assume input is already [F, 1, H, W] or [F, H, W, 1]
            if depth_tensor.shape[1] != 1 and depth_tensor.shape[3] == 1:
                # Input might be [F, H, W, 1], permute to [F, 1, H, W]
                depth_tensor = depth_tensor.permute(0, 3, 1, 2)
                logging.info("Permuted 4D depth from [F, H, W, 1] to [F, 1, H, W].")
            elif depth_tensor.shape[1] == 1:
                # Input is already [F, 1, H, W], do nothing
                logging.info("Loaded 4D depth seems to be in correct [F, 1, H, W] format.")
            else:
                # Unexpected 4D shape
                logging.error(f"Loaded depth has 4 dims but shape {depth_tensor.shape} is not recognized. Expected [F, 1, H, W] or [F, H, W, 1].")
                raise ValueError(f"Unexpected 4D depth shape: {depth_tensor.shape}")
        else:
            # Unexpected number of dimensions
            logging.error(f"Loaded depth has unexpected number of dimensions: {len(depth_tensor.shape)}. Shape: {depth_tensor.shape}")
            raise ValueError(f"Unexpected depth shape: {depth_tensor.shape}")

        # Now depth_tensor should be [F, 1, H, W]
        self.depth = torch.nn.Parameter(depth_tensor, requires_grad=True)
        logging.info(f"Depth map loaded as Parameter, final shape: {self.depth.shape}, requires_grad: {self.depth.requires_grad}")


        # Load K only if optimizing intrinsics (K_gt is handled later)
        intrinsics_save_path = os.path.join(depth_save_base, "intrinsics.npy")
        if self.opt.opt_intrinsics:
            if os.path.exists(intrinsics_save_path):
                self.K = torch.tensor(np.load(intrinsics_save_path)).float().to(self.device) # Load K if optimizing
                logging.info("Loaded K for intrinsic optimization from intrinsics.npy")
        else:
                logging.warning(f"Optimizing intrinsics but intrinsics file not found at {intrinsics_save_path}. Check paths.")
                # Rely on CameraIntrinsics initialization as fallback

        # get_track_depths uses F.grid_sample which expects [N, C, H, W] input (self.depth)
        # and [N, H_out, W_out, 2] grid (coords_normalized_static)
        # self.depth shape [F, 1, H, W] is correct for this.
        # Add a final check before calling get_track_depths
        if len(self.depth.shape) != 4 or self.depth.shape[1] != 1:
             logging.error(f"CRITICAL: self.depth shape is {self.depth.shape} before calling get_track_depths. Expected 4 dimensions [F, 1, H, W].")
             raise ValueError("Depth shape error before get_track_depths")
        self.get_track_depths()

    def load_images(self):
        image_paths_base = os.path.join(self.opt.BASE, "rgb")
        image_paths = sorted([os.path.join(image_paths_base, x) for x in os.listdir(image_paths_base) if x.endswith(".png") or x.endswith(".jpg")])

        # 【New】Truncate image path list based on --max_frames parameter
        if hasattr(self.opt, 'max_frames') and self.opt.max_frames is not None and self.opt.max_frames > 0:
            logging.info(f"Detected --max_frames parameter, limiting image count to first {self.opt.max_frames} frames.")
            image_paths = image_paths[:self.opt.max_frames]
        else:
            logging.info(f"--max_frames not set, loading all {len(image_paths)} frames.")

        self.images = []
        # Use tqdm to add progress bar for monitoring loading process
        for image_path in tqdm(image_paths, desc="Loading images"):
            im = imageio.imread(image_path)
            self.images.append(im)
        
        self.images = torch.tensor(np.array(self.images))

        _, self.height, self.width, _ = self.images.shape
        self.shape = (self.height, self.width)

        self.num_frames = self.images.shape[0]
        logging.info(f"Finally loaded {self.num_frames} frames.") # Add log to confirm final frame count

    def load_tracklets(self):
        cotracker_dir = self.opt.cotracker_path
        cotrack_results = np.load(os.path.join(self.opt.BASE, cotracker_dir, "results.npz"))
        
        # Loading all raw data from .npz file
        all_tracks_full = cotrack_results["all_tracks"]
        all_vis_full = cotrack_results["all_visibilities"]
        track_init_frames_full = cotrack_results["init_frames"]
        
        # 【Core Modification】: Create a mask to keep only points initialized within self.num_frames
        # self.num_frames is 50 here, so this mask will mark all points with initial frame number < 50
        valid_points_mask = track_init_frames_full < self.num_frames
        
        # Use this mask to filter the second dimension (N) of all arrays related to 'points'
        # Meanwhile, still use self.num_frames to truncate the first dimension (F)
        self.all_tracks = all_tracks_full[:self.num_frames, valid_points_mask, :]
        self.all_vis = all_vis_full[:self.num_frames, valid_points_mask]
        self.track_init_frames = track_init_frames_full[valid_points_mask]
        
        num_original_points = len(track_init_frames_full)
        num_valid_points = np.sum(valid_points_mask)
        logging.info(f"Tracklets filtered: Kept {num_valid_points} out of {num_original_points} points "
                    f"that were initialised within the first {self.num_frames} frames.")

        self.confidences = self.all_vis.copy().astype(np.float32)
        
        save_path = os.path.join(self.opt.BASE, cotracker_dir, "filtered_results.npz")

        self.filter_cotracker()
        np.savez_compressed(save_path, tracks=self.all_tracks, vis=self.all_vis, labels=self.all_labels)

        # Convert to tensors FIRST
        self.all_tracks = torch.tensor(self.all_tracks).float()
        self.track_init_frames = torch.tensor(self.track_init_frames).int()
        self.all_vis = torch.tensor(self.all_vis).bool()
        self.all_labels = torch.tensor(self.all_labels).int()
        self.confidences = torch.tensor(self.confidences).float()

        # THEN move essential tensors to device immediately
        # This ensures they are on the correct device before get_track_depths might be called
        self.all_tracks = self.all_tracks.to(self.device)
        self.all_vis = self.all_vis.to(self.device)
        self.track_init_frames = self.track_init_frames.to(self.device)
        self.all_labels = self.all_labels.to(self.device)
        self.confidences = self.confidences.to(self.device)

        self.all_tracks_static = self.all_tracks[:, self.all_labels==0, :]
        self.all_vis_static = self.all_vis[:, self.all_labels==0]
        self.all_tracks_static_init = self.track_init_frames[self.all_labels==0]
        self.track_init_frames_static = self.track_init_frames[self.all_labels==0] # This might be redundant with all_tracks_static_init
        self.static_confidences = self.confidences[:, self.all_labels==0]

        # Ensure we're ONLY using DeVA dynamic points (label==1)
        # Explicitly NOT using any EPI dynamic points (which would be in label==3)
        dyn_mask = self.all_labels==1
        logging.info(f"Using ONLY DeVA dynamic points (label==1) for dynamic optimization")
        logging.info(f"Found {torch.sum(dyn_mask).item()} DeVA dynamic points out of {len(self.all_labels)} total points")
        
        self.track_init_frames_dyn = self.track_init_frames[dyn_mask]
        self.all_tracks_dyn = self.all_tracks[:, dyn_mask, :]
        self.all_vis_dyn = self.all_vis[:, dyn_mask]
        self.dyn_confidences = self.confidences[:, dyn_mask]

        # Ensure derived tensors are also on device (might be redundant if base tensors are already moved, but safe)
        self.all_tracks_static = self.all_tracks_static.to(self.device)
        self.all_vis_static = self.all_vis_static.to(self.device)
        self.all_tracks_static_init = self.all_tracks_static_init.to(self.device)
        self.track_init_frames_static = self.track_init_frames_static.to(self.device)
        self.static_confidences = self.static_confidences.to(self.device)

        self.track_init_frames_dyn = self.track_init_frames_dyn.to(self.device)
        self.all_tracks_dyn = self.all_tracks_dyn.to(self.device)
        self.all_vis_dyn = self.all_vis_dyn.to(self.device)
        self.dyn_confidences = self.dyn_confidences.to(self.device)

        self.num_points_static = self.all_tracks_static.shape[1] 
        self.num_points_dyn = self.all_tracks_dyn.shape[1]

        logging.info(f"Number of static points: {self.num_points_static}")
        logging.info(f"Number of dynamic points: {self.num_points_dyn}")

    def load_data(self):
        """
        Method used to load in rgb, cotrackers, and predict / store depth from unidepth
        Ensure paths are set correctly
        """
        # Base data directory
        self.opt.BASE = os.path.abspath(os.path.join(self.opt.workdir, self.opt.video_name))
        print(f"Debug: BASE directory set to: {self.opt.BASE}")
        
        # Output directory
        self.opt.output_dir = os.path.abspath(os.path.join(self.opt.BASE, "dynamicBA", self.opt.experiment_name))
        print(f"Debug: output_dir directory set to: {self.opt.output_dir}")

        os.makedirs(self.opt.output_dir, exist_ok=True)
        
        # Log settings - ensure UTF-8 encoding is used
        if self.opt.log:
            reload(logging)
            # Modification: Explicitly specify encoding as UTF-8
            log_file = os.path.join(self.opt.output_dir, 'training_info.log')
            handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            logging.basicConfig(handlers=[handler], level=logging.INFO,
                               format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')
            print(f"Log file will be saved to: {log_file} (using UTF-8 encoding)")
        else:
            # If not logging to file, ensure at least a stream handler exists
            if not logging.getLogger().hasHandlers():
                 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')
            else:
                 # If handler exists, only adjust level
                 logging.getLogger().setLevel(logging.INFO)

        arg_dict = vars(self.opt)
        json_string = json.dumps(arg_dict, indent=4)
        logging.info(json_string)
        print(f"Configuration used: {json_string}") # Add print configuration info

        self.load_images()
        self.get_masks()
        self.load_tracklets()
        self.load_gt()
        self.load_depths()
        self.create_init_static_points()
    
    def reset_schedulers(self, patience=10):
        
        if hasattr(self, 'schedulers'):
            for scheduler in self.schedulers.values():
                del scheduler

        self.schedulers = {}

        for opt in self.optimizers:
            if opt == "shift":  # Lets not use scheduler for shift
                continue
            # Create schedulers for all optimizers, including depth and reproj_depth
            self.schedulers[opt] = ReduceLROnPlateau(self.optimizers[opt], 'min', factor=0.5, patience=patience, min_lr=1e-4)
            
            # Logging
            if opt in ["depth", "reproj_depth"]:
                logging.info(f"Created LR scheduler for {opt}, initial LR={self.optimizers[opt].param_groups[0]['lr']}")

    def init_optimization(self, lr=1e-3, patience=10):

        """
        Initializes the optimization process
        """

        if not hasattr(self, 'active_optimizers'):
            self.active_optimizers=[]

        translation_params = self.controlpoints_static.get_param()
        
        self.translation_optimizer = torch.optim.Adam([translation_params], lr=lr, weight_decay=0, amsgrad=True)

        rotation_params, translation_params = self.poses.get_rotation_and_translation_params()
        self.pose_rotation_optimizer = torch.optim.Adam(rotation_params, lr=lr, weight_decay=0, amsgrad=True)
        self.pose_translation_optimizer = torch.optim.Adam(translation_params, lr=lr, weight_decay=0, amsgrad=True)
        
        translation_dyn_params, _ = self.controlpoints_dyn.get_params()
        self.t_dyn_optimizer = torch.optim.Adam([translation_dyn_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)

        self.optimizers = {
            "points_stat_t": self.translation_optimizer,
            "rotation_pose": self.pose_rotation_optimizer,
            "translation_pose": self.pose_translation_optimizer,
            "points_dyn_t": self.t_dyn_optimizer
        }

        if self.opt.opt_intrinsics:
            self.intrinsics_optimizer = torch.optim.Adam(self.intrinsics.parameters(), lr=self.opt.intrinsics_lr, weight_decay=0, amsgrad=True)
            self.optimizers["intrinsics"]= self.intrinsics_optimizer

        # Add depth map optimizer (use separate learning rate, may need adjustment)
        # Check if self.depth exists and is a Parameter
        if hasattr(self, 'depth') and isinstance(self.depth, torch.nn.Parameter):
            depth_lr = getattr(self.opt, 'depth_lr', 1e-4) # Add --depth_lr to args or use default
            self.depth_optimizer = torch.optim.Adam([self.depth], lr=depth_lr, weight_decay=0, amsgrad=True)
            self.optimizers["depth"] = self.depth_optimizer
            logging.info(f"Added depth optimizer with lr={depth_lr}")
        else:
            # This case might happen if load_state is used and depth wasn't saved as parameter,
            # or if load_depths fails before creating the parameter.
            logging.warning("self.depth is not an optimizable Parameter, cannot create depth optimizer.")

        # --- Add reproj_depths optimizer ---
        if hasattr(self, 'reproj_depths_param') and isinstance(self.reproj_depths_param, torch.nn.Parameter):
            # Use same LR as depth or specify separately
            reproj_depth_lr = getattr(self.opt, 'reproj_depth_lr', getattr(self.opt, 'depth_lr', 1e-4))
            self.reproj_depth_optimizer = torch.optim.Adam([self.reproj_depths_param], lr=reproj_depth_lr, weight_decay=0, amsgrad=True)
            self.optimizers["reproj_depth"] = self.reproj_depth_optimizer
            logging.info(f"Added reproj_depth optimizer with lr={reproj_depth_lr}")
        else:
            logging.warning("self.reproj_depths_param not found or not a Parameter, cannot create reproj_depth optimizer.")

        self.reset_schedulers(patience)
        

    def to_device(self):
        """
        Moves everything to device, increase robustness when handling cases where certain attributes do not exist
        """
        # Processing all_tracks related attributes
        for attr_name in ['all_tracks', 'all_vis', 'confidences', 'all_labels']:
            if hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self, attr_name).to(self.device))
            else:
                logging.warning(f"Attribute {attr_name} does not exist, skipping move to device")
        
        # Processing static track related attributes
        if hasattr(self, 'all_tracks_static'):
            self.all_tracks_static = self.all_tracks_static.to(self.device)
        if hasattr(self, 'all_tracks_static_init'):
            self.all_tracks_static_init = self.all_tracks_static_init.to(self.device)
        if hasattr(self, 'static_confidences'):
            self.static_confidences = self.static_confidences.to(self.device)
        
        # Processing dynamic track related attributes
        if hasattr(self, 'all_tracks_dyn'):
            self.all_tracks_dyn = self.all_tracks_dyn.to(self.device)
        if hasattr(self, 'all_vis_dyn'):
            self.all_vis_dyn = self.all_vis_dyn.to(self.device)
        if hasattr(self, 'dyn_confidences'):
            self.dyn_confidences = self.dyn_confidences.to(self.device)
        if hasattr(self, 'all_tracks_dyn_depth') and self.all_tracks_dyn_depth is not None:
            self.all_tracks_dyn_depth = self.all_tracks_dyn_depth.to(self.device)

        # Processing track_init_frames
        if hasattr(self, 'track_init_frames'):
            self.track_init_frames = self.track_init_frames.to(self.device)
        
        # Processing static track depth
        if hasattr(self, 'all_tracks_static_depth'):
            self.all_tracks_static_depth = self.all_tracks_static_depth.to(self.device)
        # Processing depth related attributes
        if hasattr(self, 'depth'):
            if isinstance(self.depth, torch.nn.Parameter):
                if self.depth.device != self.device:
                    self.depth = torch.nn.Parameter(self.depth.data.to(self.device), requires_grad=self.depth.requires_grad)
            elif isinstance(self.depth, torch.Tensor):
                if self.depth.device != self.device:
                    self.depth = self.depth.to(self.device)

        # Processing reprojection depth related attributes
        if hasattr(self, 'reproj_depths_param') and isinstance(self.reproj_depths_param, torch.nn.Parameter):
            if self.reproj_depths_param.device != self.device:
                self.reproj_depths_param = torch.nn.Parameter(self.reproj_depths_param.data.to(self.device), requires_grad=self.reproj_depths_param.requires_grad)
        
        # Processing mask related attributes
        if hasattr(self, 'dyn_masks') and isinstance(self.dyn_masks, torch.Tensor):
            if self.dyn_masks.device != self.device:
                self.dyn_masks = self.dyn_masks.to(self.device)
                logging.info(f"Moved dyn_masks to device: {self.dyn_masks.device}")
        
        if hasattr(self, 'is_static_strict_tensor') and isinstance(self.is_static_strict_tensor, torch.Tensor):
            if self.is_static_strict_tensor.device != self.device:
                self.is_static_strict_tensor = self.is_static_strict_tensor.to(self.device)
                logging.info(f"Moved is_static_strict_tensor to device: {self.is_static_strict_tensor.device}")

        # Processing control points and poses
        if hasattr(self, 'poses'):
            self.poses = self.poses.to(self.device)
        if hasattr(self, 'controlpoints_static'):
            self.controlpoints_static = self.controlpoints_static.to(self.device)
        if hasattr(self, 'controlpoints_dyn'):
            self.controlpoints_dyn = self.controlpoints_dyn.to(self.device)

        # Processing camera intrinsics
        if hasattr(self, 'opt') and self.opt.opt_intrinsics:
            if hasattr(self, 'intrinsics'):
                self.intrinsics = self.intrinsics.to(self.device)
        else:
            if hasattr(self, 'K_gt') and self.K_gt is not None: 
                 self.K_gt = self.K_gt.to(self.device)
            if hasattr(self, 'K') and self.K is not None: 
                 self.K = self.K.to(self.device)
        
        logging.info("All available tensors moved to device.")

    def load_gt(self):

        if not self.opt.opt_intrinsics:
            intrinsic = np.load(os.path.join(self.opt.BASE, "K.npy"))
            self.K_gt = torch.tensor(intrinsic).float()

            if len(self.K_gt.shape) == 2:
                self.K_gt = self.K_gt.unsqueeze(0).repeat(self.num_frames, 1, 1)
                

    def set_all_seeds(self, seed=42, deterministic=True):
        """
        Set all seeds to make results reproducible.
        
        Args:
            seed: Integer seed for reproducibility.
            deterministic: If True, ensures PyTorch operations are deterministic.
                        May impact performance due to deterministic algorithms.
        
        Note: Setting deterministic=True may significantly impact performance.
            Only use it when absolute reproducibility is required.
        """
        # Python's random seed
        
        random.seed(seed)
        
        # NumPy seed
        np.random.seed(seed)
        
        # PyTorch seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # PyTorch backend
        if deterministic:
            logging.info("Deterministic operations enabled.")
            # Configure backend for deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set environment variable for deterministic operations
            os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8"
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Optional: Force PyTorch operations to be deterministic, warn if not possible
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                logging.info("torch.use_deterministic_algorithms(True, warn_only=True) set.")
            except AttributeError:
                logging.warning("torch.use_deterministic_algorithms is not available in this PyTorch version. Determinism might not be fully enforced.")
        else:
            logging.info("Deterministic operations disabled.")
            # Better performance, but not fully deterministic
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def log_timer(self, stage):
        timer_file = os.path.join(self.opt.output_dir, "timer.txt")
        current_time = time.time()
        if not hasattr(self, 'start_time'): 
            self.start_time = current_time 
        with open(timer_file, "a", encoding='utf-8') as f:
            f.write(f"{stage}: {current_time - self.start_time:.4f}\n")

    def initialize(self):
        """
        Initializes the engine: sets seeds, loads all data, initializes variables,
        optimization components, and moves data to the target device.
        """
        self.start_time = time.time()
        logging.info(f"Engine initialization started at {self.start_time}")

        self.set_all_seeds(seed=self.opt.seed, deterministic=self.opt.deterministic)
        logging.info(f"Random seeds set with seed={self.opt.seed}, deterministic={self.opt.deterministic}")

        self.load_data()
        logging.info("Data loading complete.")
        
        # Pre-generating and caching Co-tracker static point masks
        logging.info("Pre-generating Co-tracker static mask cache...")
        self.get_cotracker_static_mask()
        logging.info("Co-tracker static mask cache ready.")
        
        # Add diagnostics for EPI masks
        logging.info("Running EPI mask diagnostics...")
        self.check_epi_masks()
        logging.info("EPI mask diagnostics complete.")

        self.init_vars()
        logging.info("Variable initialization complete.")

        self.init_optimization(lr=getattr(self.opt, 'ba_lr', 1e-3))
        logging.info("Optimization components initialized.")

        self.to_device()
        logging.info(f"All components moved to device: {self.device}")
        
        self.log_timer("Initialization_Total") 
        logging.info("Engine initialization complete.")

    def freeze_frame(self, frame_index):

        pos_param = getattr(self.poses, f'delta_rotation_{frame_index}')
        pos_param.requires_grad = False
        trans_param = getattr(
            self.poses, f'delta_translation_{frame_index}')
        trans_param.requires_grad = False

    def init_pose(self, frame_index):  # initialize with previous frame
        so3_1, tr_1 = self.poses.get_raw_value(frame_index-1)
        self.poses.set_rotation_and_translation(frame_index, so3_1, tr_1)


    def smooth_reg(self, filter_outlier=False, debug=False):
        """
        Temporal smoothness and spatial smoothness (neighbors)
        """
        # Get trajectories and visibility of dynamic points
        ts, _ = self.controlpoints_dyn.forward() # -> for ts: N x F x 3
        vis_dyn = self.all_vis_dyn

        # Calculate displacement between adjacent frames
        d = ts[:,1:,:] - ts[:,:-1,:]
        d_norm = torch.norm(d, dim=-1, p=2)

        # Calculate visibility mask
        mask = torch.logical_and(vis_dyn[1:], vis_dyn[:-1]).T  # visibility mask
        
        # Debug info
        if debug:
            mask_sum = torch.sum(mask).item()
            mask_size = mask.numel()
            print(f"[smooth_reg] Mask size: {mask_size}, True values: {mask_sum} ({mask_sum/mask_size*100:.2f}%)")
            
            if mask_sum > 0:
                valid_d_norm = d_norm[mask]
                print(f"[smooth_reg] Valid displacement stats - Min: {torch.min(valid_d_norm).item():.6f}, Max: {torch.max(valid_d_norm).item():.6f}, Mean: {torch.mean(valid_d_norm).item():.6f}")
            else:
                print("[smooth_reg] No valid point pairs to calculate displacement")

        outliers = None

        if filter_outlier:
            d_norm[~mask] = 0.0
            outliers = d_norm

        # Handling case where mask is all False
        if torch.sum(mask) == 0:
            if debug:
                print("[smooth_reg] Warning: No True values in mask, returning zero loss")
            return torch.tensor(0.0, device=self.device), None, outliers

        reg_t = torch.mean(d_norm[mask])

        return reg_t, None, outliers

    def ordered_ratio(self, disp_a, disp_b):
        ratio_a = torch.maximum(disp_a, disp_b) / \
            (torch.minimum(disp_a, disp_b)+1e-5)
        return ratio_a - 1

    def get_intrinsics_K(self, idxes=None):

        if self.opt.opt_intrinsics:
            if idxes is None:
                return self.intrinsics.get_K().unsqueeze(0)
            else:
                return self.intrinsics.get_K().unsqueeze(0).repeat(len(idxes), 1, 1)
        else:
            if idxes is None:
                return self.K_gt
            else:
                # Check dimensions of K_gt, if it is just a single 3x3 matrix, need to copy for all frames
                if len(self.K_gt.shape) == 2:  # 3x3 matrix
                    return self.K_gt.unsqueeze(0).repeat(len(idxes), 1, 1)
                else:  # Already a multi-frame tensor
                    return self.K_gt[idxes]

    def fuse_helper_deva(self, R, t, depth, label, depthmap_res, f, coords, src_ray_homo, region_mask):
        """
        Fusion method specifically for processing dyn_masks_filters_deva format masks
        
        Args:
            R, t: Camera extrinsics
            depth: Current frame depth map
            label: Region label (0=Static, 1=DeVA Dynamic)
            depthmap_res: Result depth map
            f: Frame index
            coords: Pixel coordinates
            src_ray_homo: Ray direction
            region_mask: Numpy boolean mask, identifying the region currently being processed
        """
        if label == 0:
            # Static region - Non-DeVA region (background/static region)
            vis = self.all_vis_static[f]             # N
            track_sel = torch.where(vis)[0]
            control_pts_3D_world = self.controlpoints_static(track_sel)        # N x 3  -->  static points
        elif label == 1:
            # DeVA dynamic region
            track_dyn = self.all_tracks_dyn[f]
            vis = self.all_vis_dyn[f]
            track_dyn = track_dyn[vis]
            track_sel_dyn = torch.where(vis)[0]
            control_pts, _  = self.controlpoints_dyn(frames=f, points=track_sel_dyn)    # N x 1 x 3
            control_pts_3D_world = control_pts.squeeze(1)                                    # N x 3
        else:
            # Other regions (boundary, etc.), do not interpolate
            logging.debug(f"fuse_helper_deva: Unknown label {label}, skipping interpolation")
            depthmap_res[region_mask] = depth[region_mask]
            return

        # Convert numpy mask to torch tensor
        if isinstance(region_mask, np.ndarray):
            mask_tensor = torch.from_numpy(region_mask).to(self.device)
        else:
            mask_tensor = region_mask.to(self.device)

        if len(control_pts_3D_world) == 0:
            depthmap_res[mask_tensor] = depth[mask_tensor]     # no alignment if we have no control points
            logging.debug(f"fuse_helper_deva: Frame {f} label {label} has no available control points, keeping original depth")
            return
    
        K = self.get_intrinsics_K([f])[0]
        control_pts_2D, control_points_depth = self.project(R, t, K, control_pts_3D_world)
        
        in_range = torch.logical_and(
            torch.logical_and(control_pts_2D[:,0] >= 0, control_pts_2D[:,0] < self.shape[1]), 
            torch.logical_and(control_pts_2D[:,1] >= 0, control_pts_2D[:,1] < self.shape[0])
        )   # filter in-view after projection
        in_range = torch.logical_and(in_range, control_points_depth > 0)

        control_pts_2D = control_pts_2D[in_range]
        control_points_depth = control_points_depth[in_range]
        
        if torch.sum(in_range) == 0:
            depthmap_res[mask_tensor] = depth[mask_tensor]     # no alignment if we have no control points
            logging.debug(f"fuse_helper_deva: Frame {f} label {label} has no control points in view, keeping original depth")
            return
            
        unidepth_vals = self.interpolate(depth.unsqueeze(-1), control_pts_2D).squeeze(-1)
        
        depth_src = depth[coords[:,1].long(), coords[:,0].long()]

        points_3d_src = (src_ray_homo * depth_src.unsqueeze(-1))   # N x 3
        points_3d_world = (R @ points_3d_src.T + t).T              # 3d reprojection of unidepth for this frame, used to find knn

        points_3d_world_grid = points_3d_world.reshape((self.shape[0], self.shape[1], 3))
        control_pts_3D_world_unidepth = points_3d_world_grid[control_pts_2D[:, 1].long(), control_pts_2D[:,0].long()]  # axis seems to be flipped

        points_3d_query = points_3d_world[mask_tensor.flatten()]

        if len(points_3d_query) == 0:
            logging.debug(f"fuse_helper_deva: Frame {f} label {label} has no query points, skipping")
            return

        knn_result = knn_points(points_3d_query.unsqueeze(0), control_pts_3D_world_unidepth.unsqueeze(0),  K=3)   # we find nn using reproject unidepth 3D instead

        idx = knn_result.idx[0]  # Shape: (num_points1, k)
        dists = knn_result.dists[0]

        weights = 1 / (dists + 1e-8) 

        weights_sum = weights.sum(dim=-1, keepdim=True)
        probabilities = weights / (weights_sum)
        scale = control_points_depth / unidepth_vals

        scales = torch.sum(scale[idx] * probabilities, axis=1)
    
        depthmap_res[mask_tensor] = depth[mask_tensor] * scales

        if torch.sum(scales < 0) > 0:
            logging.warning(f"fuse_helper_deva: Frame {f} label {label} has negative scale factors")
            breakpoint()
            
        # Log interpolation statistics
        if f < 5:  # Log detailed info for the first few frames
            avg_scale = scales.mean().item()
            min_scale = scales.min().item()
            max_scale = scales.max().item()
            region_name = "Static" if label == 0 else "DeVA Dynamic"
            logging.debug(f"fuse_helper_deva: Frame {f} {region_name} region - Scale factor stats: Mean={avg_scale:.3f}, Range=[{min_scale:.3f}, {max_scale:.3f}], Interpolated points={len(scales)}")
        elif f % 20 == 0:  # Log summary info every 20 frames
            avg_scale = scales.mean().item()
            region_name = "Static" if label == 0 else "DeVA Dynamic"
            logging.debug(f"fuse_helper_deva: Frame {f} {region_name} region interpolation completed (Mean scale={avg_scale:.3f}, Interpolated points={len(scales)})")

    def get_masks(self):
        """
        Loads masks from DeVA and optionally EPI directories.
        Processes them into:
        - self.dyn_masks: Instance IDs based on DeVA (for consistency tracking).
        - self.dyn_masks_filters: Filtered mask with labels:
            0 -> static (core, not EPI dynamic)
            1 -> dynamic (DeVA core, not EPI dynamic)
            2 -> rejected/boundary (not EPI dynamic)
            3 -> epi_dynamic (highest priority)
        Also performs erosion on static and dynamic masks for better boundaries.
        """
        # --- Paths for DeVA and EPI masks ---
        if hasattr(self.opt, 'mask_name') and self.opt.mask_name is not None:
            mask_name = self.opt.mask_name
        else:
            mask_name = 'Annotations'
        deva_anot_path = os.path.join(self.opt.BASE, self.opt.dyn_mask_dir, mask_name)
        epi_mask_path = os.path.join(self.opt.BASE, "epi_mask_11111") 
        
        # Check if DeVA path exists
        if not os.path.exists(deva_anot_path):
            logging.error(f"DeVA annotation path not found: {deva_anot_path}")
            h, w = self.shape
            num_f = self.num_frames
            self.dyn_masks = torch.zeros((num_f, h, w), dtype=torch.int64) 
            self.dyn_masks_filters = (torch.zeros((num_f, h, w), dtype=torch.uint8) + 2).numpy() # All rejected
            logging.warning("DeVA path missing, created default empty/rejected masks.")
            return

        deva_mask_files_full = sorted([f for f in os.listdir(deva_anot_path) if f.endswith((".npy", ".png"))])
    
        # [Core Modification]: Truncate the mask file list based on self.num_frames
        # Note: self.num_frames should have been set by load_images() at this point
        deva_mask_files = deva_mask_files_full[:self.num_frames]
        logging.info(f"Mask files truncated to {len(deva_mask_files)} frames.")

        all_masks_deva = []
        all_masks_epi = []
        
        logging.info(f"Loading masks from DeVA: {deva_anot_path}")
        logging.info(f"Attempting to load masks from EPI: {epi_mask_path}")

        # [Note] The loop variable deva_mask_file here is already the truncated list
        for f_idx, deva_mask_file in enumerate(deva_mask_files):    
            # Load DeVA mask
            try:
                deva_full_path = os.path.join(deva_anot_path, deva_mask_file)
                if deva_mask_file.endswith(".npy"):
                    mask_dyn = np.load(deva_full_path)
                elif deva_mask_file.endswith(".png"):
                    mask_dyn = imageio.imread(deva_full_path)
                    if len(mask_dyn.shape) == 3: # Handle color index png
                        mask_dyn = mask_dyn.astype(np.int64)
                        mask_dyn = mask_dyn[:,:,0] + mask_dyn[:,:,1] * 256 + mask_dyn[:,:,2] * 256 * 256
                all_masks_deva.append(mask_dyn)
            except Exception as e:
                logging.error(f"Failed to load or process DeVA mask {deva_mask_file}: {e}")
                h, w = self.shape 
                all_masks_deva.append(np.zeros((h, w), dtype=np.int64))

            # Load corresponding EPI mask
            # Instead of using DeVA mask filename as base, use frame index to construct EPI mask filename
            # Original approach (wrong):
            # base_name, _ = os.path.splitext(deva_mask_file)
            # epi_mask_filename_png = f"{base_name}.png"
            
            # Get the frame index from the current loop
            frame_index = f_idx + 1  # 1-indexed to match file naming scheme
            epi_mask_filename_png = f"frame_{frame_index:04d}_dynamic_mask.png"
            epi_full_path = os.path.join(epi_mask_path, epi_mask_filename_png)
            
            logging.info(f"Looking for EPI mask at: {epi_full_path}")
            epi_mask = None
            
            # Check if EPI mask directory exists
            if not os.path.exists(epi_mask_path):
                # Warning already logged earlier
                pass
            # Check if specific EPI mask file exists
            elif not os.path.exists(epi_full_path):
                logging.debug(f"EPI mask file not found: {epi_full_path}")
                
                # Try alternative pattern as fallback
                alt_file_pattern = f"*{frame_index:04d}*.png"
                alt_files = glob.glob(os.path.join(epi_mask_path, alt_file_pattern))
                if alt_files:
                    logging.info(f"Found alternative EPI mask files matching pattern '{alt_file_pattern}': {alt_files}")
                    epi_full_path = alt_files[0]  # Use first match
            
            if os.path.exists(epi_full_path):
                try:
                    # Load the EPI mask file
                    epi_mask_raw = imageio.imread(epi_full_path)
                    
                    # Add more detailed debugging information about the loaded mask
                    unique_values = np.unique(epi_mask_raw)
                    min_val = np.min(epi_mask_raw)
                    max_val = np.max(epi_mask_raw)
                    logging.info(f"Loaded EPI mask: {epi_full_path}")
                    logging.info(f"  Shape: {epi_mask_raw.shape}, dtype: {epi_mask_raw.dtype}")
                    logging.info(f"  Value range: min={min_val}, max={max_val}")
                    logging.info(f"  Unique values: {unique_values}")
                    
                    # Handle multi-channel images
                    if len(epi_mask_raw.shape) == 3:
                        logging.info(f"  Converting 3-channel mask to single channel")
                        # Check all channels for non-zero values
                        channel_sums = [np.sum(epi_mask_raw[..., i]) for i in range(epi_mask_raw.shape[2])]
                        logging.info(f"  Channel sums: {channel_sums}")
                        
                        # If all channels are identical or only one has data, use first channel
                        epi_mask_raw = epi_mask_raw[..., 0]
                        logging.info(f"  After conversion: shape={epi_mask_raw.shape}, unique values={np.unique(epi_mask_raw)}")
                    
                    # Check if mask has any non-zero values
                    if np.sum(epi_mask_raw) == 0:
                        logging.warning(f"⚠️ EPI mask is all zeros/black: {epi_full_path}")
                        
                        # Try checking image format and properties
                        if hasattr(imageio, 'get_reader'):
                            try:
                                reader = imageio.get_reader(epi_full_path)
                                meta = reader.get_meta_data()
                                logging.info(f"  Image metadata: {meta}")
                            except Exception as e:
                                logging.warning(f"  Could not read metadata: {e}")
                    
                    # Convert to boolean mask, with possible value adjustment
                    # If the mask values are very low (e.g., 0-1 range), amplify them
                    if max_val > 0 and max_val < 10:
                        # Assuming this is a grayscale mask with low values
                        logging.info(f"  Mask has low values, applying amplification")
                        epi_mask_raw = (epi_mask_raw * (255.0 / max_val)).astype(np.uint8)
                        logging.info(f"  After amplification: min={np.min(epi_mask_raw)}, max={np.max(epi_mask_raw)}")
                    
                    # Try different thresholds if the mask is all zeros
                    if np.sum(epi_mask_raw > 0) == 0:
                        for threshold in [0, 10, 25, 50]:
                            non_zero = np.sum(epi_mask_raw > threshold)
                            logging.info(f"  Pixels > {threshold}: {non_zero} ({non_zero/(epi_mask_raw.size)*100:.2f}%)")
                    
                    # Create final boolean mask
                    threshold = 0  # Default threshold
                    epi_mask = (epi_mask_raw > threshold).astype(bool)
                    non_zero = np.sum(epi_mask)
                    logging.info(f"  Final mask has {non_zero} non-zero pixels ({non_zero/(epi_mask.size)*100:.2f}%)")
                    
                    # Check shape matching
                    if epi_mask.shape != all_masks_deva[-1].shape:
                        logging.warning(f"EPI mask shape mismatch: {epi_mask_filename_png}, EPI shape: {epi_mask.shape}, DeVA shape: {all_masks_deva[-1].shape}. Skipping this mask.")
                        epi_mask = None 
                except Exception as e:
                    logging.warning(f"Error loading or processing EPI mask {epi_mask_filename_png}: {e}. Skipping this mask.")
                    epi_mask = None

            all_masks_epi.append(epi_mask) 

        # --- Process DeVA masks for consistent instance IDs ---
        if not all_masks_deva:
             logging.error("No DeVA masks were loaded.")
             h, w = self.shape
             num_f = self.num_frames
             self.dyn_masks = torch.zeros((num_f, h, w), dtype=torch.int64)
             self.dyn_masks_filters = (torch.zeros((num_f, h, w), dtype=torch.uint8) + 2).numpy() # All rejected
             logging.warning("No DeVA masks loaded, cannot generate filters.")
             return

        masks_deva_np = np.array(all_masks_deva)
        if masks_deva_np.size == 0: # Check if the array is empty after creation
            logging.error("Final DeVA numpy array is empty after attempting to load masks.")
        else:
            # Only process unique IDs if masks_deva_np_array is not empty
            _, unique_deva_ids_np = np.unique(masks_deva_np, return_inverse=True)
            self.dyn_masks = torch.tensor(unique_deva_ids_np.reshape(masks_deva_np.shape), dtype=torch.int64, device='cpu')

        # --- Generate the final filtered mask (self.dyn_masks_filters) ---
        dyn_masks_filters_list = []
        # New: Add a separate list to store only DeVA masks without EPI influence
        dyn_masks_filters_deva_list = []
        logging.info("Generating final filtered masks (Static=0, DeVA Dyn=1, Rejected=2, EPI Dyn=3 - EPI has highest priority)...")
        
        num_loaded_frames = masks_deva_np.shape[0] 
        
        for f in range(num_loaded_frames):
            mask_dyn = masks_deva_np[f] 
            epi_mask = all_masks_epi[f]  

            # Initialize filter mask: 2 = rejected/boundary
            mask_dyn_pro = np.full(mask_dyn.shape, 2, dtype=np.uint8) 

            # Erode static region (DeVA background ID 0)
            static_mask_eroded = cv2.erode((mask_dyn == 0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=3).astype(np.bool_)
            mask_dyn_pro[static_mask_eroded] = 0 # Label core static first

            # Erode dynamic region (DeVA non-background IDs)
            dyn_mask_eroded = cv2.erode((mask_dyn != 0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=2).astype(np.bool_)
            
            # Label core DeVA dynamic ONLY if it wasn't already marked as core static
            deva_dyn_final_mask = dyn_mask_eroded & (mask_dyn_pro != 0)
            mask_dyn_pro[deva_dyn_final_mask] = 1 
            
            # Create a copy of the mask at this point before EPI is applied
            # This preserves all original DeVA dynamic regions (label=1)
            mask_dyn_deva_only = mask_dyn_pro.copy()
            dyn_masks_filters_deva_list.append(mask_dyn_deva_only)
            
            # Apply EPI dynamic mask, overwriting any previous label
            if epi_mask is not None and epi_mask.shape == mask_dyn_pro.shape:
                epi_pixels = np.sum(epi_mask)
                mask_dyn_pro[epi_mask] = 3 # EPI dynamic overwrites DeVA dynamic and static
                logging.info(f"Frame {f}: Applied EPI mask, {epi_pixels} pixels marked as EPI dynamic (label 3)")
                
                # Count pixels that were changed from DeVA dynamic (1) to EPI dynamic (3)
                deva_to_epi = np.sum((mask_dyn_deva_only == 1) & epi_mask)
                if deva_to_epi > 0:
                    logging.warning(f"Frame {f}: {deva_to_epi} pixels changed from DeVA dynamic to EPI dynamic")
                
                # Statistics for different label types
                unique_labels, counts = np.unique(mask_dyn_pro, return_counts=True)
                label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
                logging.debug(f"Frame {f} label statistics: {label_stats}")
            else:
                if epi_mask is not None:
                    logging.warning(f"Frame {f}: EPI mask exists but shape mismatch, cannot apply. EPI shape: {epi_mask.shape}, required shape: {mask_dyn_pro.shape}")
                # No EPI mask for this frame or shape mismatch, log existing labels
                unique_labels, counts = np.unique(mask_dyn_pro, return_counts=True)
                label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
                logging.debug(f"Frame {f} label statistics (no EPI): {label_stats}")

            dyn_masks_filters_list.append(mask_dyn_pro)

        # Assign the final combined result to self.dyn_masks_filters
        self.dyn_masks_filters = np.array(dyn_masks_filters_list).astype(np.uint8)
        
        # Assign the DeVA-only masks (without EPI influence) to a new attribute
        self.dyn_masks_filters_deva = np.array(dyn_masks_filters_deva_list).astype(np.uint8)
        logging.info("Created separate DeVA-only mask (dyn_masks_filters_deva) for dynamic optimization")
        
        # Log statistics comparing the two mask types
        total_deva_dynamic_original = np.sum(self.dyn_masks_filters_deva == 1)
        total_deva_dynamic_after_epi = np.sum(self.dyn_masks_filters == 1)
        total_epi_dynamic = np.sum(self.dyn_masks_filters == 3)
        
        logging.info(f"Mask statistics: Original DeVA dynamic pixels: {total_deva_dynamic_original}")
        logging.info(f"Mask statistics: DeVA dynamic pixels after EPI applied: {total_deva_dynamic_after_epi}")
        logging.info(f"Mask statistics: EPI dynamic pixels: {total_epi_dynamic}")
        logging.info(f"Mask statistics: Pixels changed from DeVA to EPI: {total_deva_dynamic_original - total_deva_dynamic_after_epi}")
        
        # Remove other older filter attributes if they exist
        if hasattr(self, 'dyn_masks_filters_epi'): delattr(self, 'dyn_masks_filters_epi')
        if hasattr(self, 'dyn_masks_filters_combined'): delattr(self, 'dyn_masks_filters_combined')

        logging.info("Finished generating final filtered masks (self.dyn_masks_filters and self.dyn_masks_filters_deva).")

        # Now create is_static_strict and is_static_strict_tensor based on dyn_masks_filters
        h, w = self.shape
        num_f_effective = self.dyn_masks_filters.shape[0] # Use effective number of frames
        self.is_static_strict = np.zeros((num_f_effective, h, w), dtype=bool)
        for f_idx in range(num_f_effective):
            # Modification: Only regions with label 0 are strictly static regions
            # Label 1=DeVA dynamic, Label 2=Boundary/rejected region, Label 3=EPI dynamic
            self.is_static_strict[f_idx] = (self.dyn_masks_filters[f_idx] == 0)
            
            # Count pixels for each label for debugging
            label_counts = np.bincount(self.dyn_masks_filters[f_idx].flatten(), minlength=4)
            logging.debug(f"Frame {f_idx} label statistics: Static(0)={label_counts[0]}, DeVA Dynamic(1)={label_counts[1]}, Boundary/Rejected(2)={label_counts[2]}, EPI Dynamic(3)={label_counts[3]}")

        if hasattr(self, 'is_static_strict') and self.is_static_strict.size > 0:
            self.is_static_strict_tensor = torch.from_numpy(self.is_static_strict).bool().to(self.device)
            logging.info(f"Created/Updated is_static_strict_tensor based on dyn_masks_filters == 0, shape: {self.is_static_strict_tensor.shape}")
            
            # Add more detailed log output showing the proportion of static regions
            total_pixels = self.is_static_strict.size
            static_pixels = np.sum(self.is_static_strict)
            logging.info(f"Strict static region proportion: {static_pixels}/{total_pixels} ({static_pixels/total_pixels*100:.2f}%)")
        elif hasattr(self, 'is_static_strict_tensor'): 
            del self.is_static_strict_tensor
        
        logging.info("Finished generating masks and is_static_strict_tensor.")


    def interpolate(self, input, uv_orig, clone=True):

        """
        takes in an input, and uv
        Interpolate values at uv from input
        input: H x W x C
        uv: N x 2

        returns output: N x C
        """
        if clone:
            uv = uv_orig.clone()   
        else:
            uv = uv_orig                                         # clones uv to avoid changing original uv
        uv[:, 0] = (uv[:, 0] / (self.width-1)) * 2 - 1
        uv[:, 1] = (uv[:, 1] / (self.height-1)) * 2 - 1

        uv = uv.unsqueeze(0).unsqueeze(0)

        input = input.unsqueeze(0).permute(0, 3, 1, 2)

        # --- Added: Ensure input and grid are on the same device ---
        input_on_device = input.to(self.device)
        uv_on_device = uv.to(self.device)
        # --- End modification ---

        output = F.grid_sample(input_on_device, uv_on_device, align_corners=True).squeeze(0).squeeze(1).T

        return output
        
    def BA_fast(self, is_dyn=False):

        """
        Parallel and faster implementation
        """

        if is_dyn:
            cp_world, _ = self.controlpoints_dyn()
            cp_world = cp_world.permute(1,0,2)
            vis_mask = self.all_vis_dyn
            gt_track = self.all_tracks_dyn
        else:
            cp_world = self.controlpoints_static().unsqueeze(0)     # 1 x N x 3
            vis_mask = self.all_vis_static
            gt_track = self.all_tracks_static
            
        # Ensure all tensors are on the same device
        device = self.device
        cp_world = cp_world.to(device)
        vis_mask = vis_mask.to(device)
        gt_track = gt_track.to(device)

        Rs, ts = self.get_poses(torch.arange(self.num_frames).to(device))    # F x 3 x 3, F x 3 x 1
        K = self.get_intrinsics_K(torch.arange(self.num_frames).to(device))  # 3x3
        
        # Ensure all tensors are on the same device
        Rs = Rs.to(device)
        ts = ts.to(device)
        K = K.to(device)

        cp_cam = torch.einsum("bin,bmi->bmn", Rs, cp_world - ts.permute(0,2,1))

        cp_coord = torch.einsum("bji,bni->bnj", K, cp_cam)
        cp_coord = cp_coord / (cp_coord[:,:,-1].unsqueeze(-1)+1e-16)
        cp_coord = cp_coord[:,:,:2]                                               # F x N x 2

        flow_error = flow_norm(gt_track - cp_coord, self.flow_norm_l1)   # F x N
        flow_error = torch.sum(flow_error * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-16)   # F
        flow_error = torch.mean(flow_error)

        return flow_error

    @torch.no_grad()
    def init_BA(self):

        """
        Initialize controlpoints by taking average of current pose and depth
        """

        control_pts_init = torch.zeros((self.num_points_static, 3)).to(self.device)
        all_conf = self.static_confidences

        for f in range(self.num_frames):
            
            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]

            track = self.all_tracks_static[f]
            vis = self.all_vis_static[f]
            conf = all_conf[f]
            track_sel = torch.where(vis)[0]                 # index of selected tracks

            N = len(track_sel)
            K = self.get_intrinsics_K([f])[0]

            control_pts_2d = track[track_sel]
            conf = conf[track_sel]

            control_pts_2d_homo = torch.concatenate((control_pts_2d, torch.ones((N,1)).float().to(control_pts_2d.device)), dim=1)  # N x 3

            src_ray =  (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.get_depth(f)
            depth_src = depth[control_pts_2d[:,1].long(), control_pts_2d[:,0].long()]

            points_3d_src = (src_ray_homo * depth_src.unsqueeze(-1))   # N x 3

            points_3d_world = (R @ points_3d_src.T + t).T
            
            control_pts_init[track_sel] += (points_3d_world * conf[:,None])

        control_pts_init = control_pts_init / (torch.sum(all_conf+1e-16, axis=0).unsqueeze(-1))

        self.controlpoints_static.set_translation(torch.arange(self.num_points_static), control_pts_init.float())

        return control_pts_init


    def _compute_parallax(self, tracks, visibilities, min_frames=4):
        """
        Calculate the maximum parallax (maximum pixel distance) for each track.
        
        Args:
            tracks (np.array): Track point coordinates, shape [F, N, 2]
            visibilities (np.array): Visibility matrix, shape [F, N]
            min_frames (int): Minimum visible frames required to calculate parallax
            
        Returns:
            np.array: Maximum parallax for each track point, shape [N,]
        """
        num_frames, num_points, _ = tracks.shape
        max_parallax = np.zeros(num_points)

        # Iterate through each point
        for i in range(num_points):
            # Get visibility mask for this point
            visible_mask = visibilities[:, i]
            
            # If there are too few visible frames, the parallax is 0
            if np.sum(visible_mask) < min_frames:
                continue
                
            visible_coords = tracks[visible_mask, i, :] # [num_visible, 2]
            
            # Calculate distances between all pairs of visible point coordinates
            # Efficient calculation using broadcasting
            diff = visible_coords[:, np.newaxis, :] - visible_coords[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=-1))
            
            # The maximum distance is the maximum parallax
            max_parallax[i] = np.max(distances)
            
        return max_parallax

    def filter_cotracker(self):
        """
        Filter points according to visibility and oob values
        Fully vectorized implementation matching the behavior of the original function
        """
        h, w = self.height, self.width

        # Check if points are within image boundaries
        within_bound = np.logical_and(self.all_tracks[:,:,1] >= 0, self.all_tracks[:,:,1] < h)
        within_bound = np.logical_and(within_bound, self.all_tracks[:,:,0] >= 0)
        within_bound = np.logical_and(within_bound, self.all_tracks[:,:,0] < w)
        within_bound = np.logical_and(within_bound, self.all_vis == 1)

        # Clip coordinates to image boundaries
        self.all_tracks[:,:,0] = np.clip(self.all_tracks[:,:,0], 0, w-1)
        self.all_tracks[:,:,1] = np.clip(self.all_tracks[:,:,1], 0, h-1)

        # Update visibility based on boundary check
        self.all_vis = np.logical_and(self.all_vis, within_bound)

        # Get masks
        # Using the DeVA-only mask (unaffected by EPI) to ensure robust classification
        if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
            logging.info("`filter_cotracker`: Classifying tracklets using EPI-unaffected `dyn_masks_filters_deva`.")
            source_mask = self.dyn_masks_filters_deva
        else:
            logging.warning("`filter_cotracker`: `dyn_masks_filters_deva` not found. Falling back to `dyn_masks_filters`.")
            logging.warning("Point classification may be affected by EPI masks.")
            source_mask = self.dyn_masks_filters

        static_mask = source_mask == 0
        dyn_mask = source_mask == 1
        invalid_mask = source_mask == 2 # Boundary/rejected region in DeVA mask

        frames, N, _ = self.all_tracks.shape
        
        # Create indices for all frames and tracks
        frame_indices = np.arange(frames).reshape(frames, 1).repeat(N, axis=1)
        
        # Get and clamp track coordinates
        x_coords = self.all_tracks[..., 0].astype(np.int64)
        y_coords = self.all_tracks[..., 1].astype(np.int64)
        
        # Create base visibility mask (visible AND not invalid)
        base_visibility = np.copy(self.all_vis)
        
        # Check which points are in invalid regions
        is_invalid = invalid_mask[frame_indices, y_coords, x_coords]
        
        # Combine visibility with invalid check - THIS IS THE KEY DIFFERENCE
        # In the original function, invalid points are excluded from visibility check during scoring
        effective_visibility = np.logical_and(base_visibility, ~is_invalid)
        
        # Get static/dynamic scores for each track
        is_static = static_mask[frame_indices, y_coords, x_coords]
        is_dynamic = dyn_mask[frame_indices, y_coords, x_coords]
        
        # Only count points that are effectively visible
        visible_static = np.logical_and(is_static, effective_visibility)
        visible_dynamic = np.logical_and(is_dynamic, effective_visibility)
        
        # Calculate totals per track
        total_visible = np.sum(effective_visibility, axis=0)
        total_static = np.sum(visible_static, axis=0)
        total_dynamic = np.sum(visible_dynamic, axis=0)
        
        # Initialize all labels as outliers (3)
        all_labels = np.full(N, 3)
        
        # Identify tracks by type matching the logic in the original function
        is_fully_static = total_static >= total_visible
        is_fully_dynamic = total_dynamic == total_visible
        is_long_enough = total_visible >= (0.2 * frames)
        
        # Apply classification rules
        all_labels[np.logical_and(is_fully_static, is_long_enough)] = 0  # Long static tracks
        all_labels[np.logical_and(is_fully_static, ~is_long_enough)] = 2  # Short static tracks
        all_labels[is_fully_dynamic] = 1  # Dynamic tracks
        # Anything not matching these conditions remains as outlier (3)
        
        self.all_labels = all_labels
        
        assert(len(self.all_labels) == self.all_tracks.shape[1])

    def set_lr(self, lr, optimizers):

        for o in self.active_optimizers:
            if o in optimizers:
                for param_group in self.optimizers[o].param_groups:
                    param_group['lr'] = lr 

    def cosine_schedule(self, t, lr_start, lr_end):
        assert 0 <= t <= 1
        return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2
    
    def rot_to_angle(self, rots):

        batch_trace = torch.vmap(torch.trace)(rots)

        batch_trace = torch.clamp((batch_trace - 1) / 2, -1+1e-4, 1-1e-4) # setting it to 1 leads to gradient issues

        batch_trace = torch.acos(batch_trace)

        return batch_trace

    def calc_pose_smooth_loss(self, ids=None):

        if ids == None:
            ids = [x for x in range(self.num_frames)]

        ids = sorted(ids)

        assert((max(ids) - min(ids)) == (len(ids) - 1)) # ids should be consecutive
        
        Rs, ts = self.poses(ids)

        v1 = ts[1:-1,:,:] - ts[:-2,:,:]
        v2 = ts[2:,:,:] - ts[1:-1,:,:]

        v1_norm = torch.norm(v1, dim=1, p=2)
        v2_norm = torch.norm(v2, dim=1, p=2)
        v_diff = v2 - v1

        v_avg_norm = (v1_norm + v2_norm) / 2
        v_diff_norm = torch.norm(v_diff, dim=1, p=2)

        reg_t = torch.mean(v_diff_norm / (v_avg_norm + 1e-12))   # exp to stretch and -1 to make it to 0

        R1 = torch.bmm(Rs[1:-1, :, :], torch.transpose(Rs[:-2, :, :], 1, 2))
        R2 = torch.bmm(Rs[2:, :, :], torch.transpose(Rs[1:-1, :, :], 1, 2))

        R_diff = torch.bmm(R2, torch.transpose(R1, 1, 2))

        ang_avg = (self.rot_to_angle(R1) + self.rot_to_angle(R2)) / 2
        ang_diff = self.rot_to_angle(R_diff)

        ang_diff[ang_diff < 0.02] = 0          # due to clamping issues, we just round to 0 if ang diff is really small

        reg_R = torch.mean((ang_diff / (ang_avg + 1e-12))) # exp to stretch and -1 to make it to 0

        return reg_t, reg_R

    def optimize_BA(self):

        """
        Performs bundle alignment over all frames. Note this is not frame pair wise loss
        """
        # Add call to visualize BA optimization regions
        self.visualize_BA_regions()
        
        self.reset_optimizer(lr=self.opt.ba_lr, patience=20)

        self.init_BA()

        logging.info("Starting optimization for BA")

        self.active_optimizers = ["rotation_pose", "translation_pose", "intrinsics", "points_stat_t"]
            
        if not self.opt.opt_intrinsics:
            self.active_optimizers.remove("intrinsics")


        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)

        earlystopper = EarlyStopper(patience=100)

        if len(self.active_optimizers) !=  0:
        
            for epoch in range(self.opt.num_BA_epochs):

                for o in self.active_optimizers:
                    self.optimizers[o].zero_grad()

                total_reproj_loss = self.BA_fast()

                total_reproj_loss *= self.opt.reproj_weight

                total = total_reproj_loss

                if self.opt.pose_smooth_weight_t > 0:
                    loss_pose_smooth_t, loss_poss_smooth_R = self.calc_pose_smooth_loss()
                    loss_pose_smooth_t = loss_pose_smooth_t * self.opt.pose_smooth_weight_t
                    loss_pose_smooth_R = loss_poss_smooth_R * self.opt.pose_smooth_weight_r
                else:
                    loss_pose_smooth_t = torch.tensor(0)
                    loss_pose_smooth_R = torch.tensor(0)

                total += (loss_pose_smooth_t + loss_pose_smooth_R)
                total.backward(retain_graph=False)

                if earlystopper.early_stop(total):
                    break

                for o in self.active_optimizers:
                    self.optimizers[o].step()
                    if o in self.schedulers:
                        self.schedulers[o].step(total)

                if epoch % self.opt.print_every == 0:
                    logging.info(f"Currently at epoch {epoch} out of {self.opt.num_BA_epochs} with pose smooth loss t: {loss_pose_smooth_t.item()}, pose smooth loss R: {loss_pose_smooth_R.item()}, reproj loss: {total_reproj_loss.item()}")
                    logging.info(f"Total loss: {total.item()} with lr: {self.schedulers['translation_pose'].get_last_lr()}")
                    # logging.info(f"Total loss: {total.item()}")

                torch.cuda.empty_cache()


    def laplacian_reg(self, outliers=False, debug=False):
        """
        Apply laplacian rigidity regularization
        We simply want the distance to neighbors to be consistent
        """
        control_pts, _ = self.controlpoints_dyn.forward()
        vis_dyn = self.all_vis_dyn.T
        neighbors_idx = self.all_neighbors 

        neighbors_loc = control_pts[neighbors_idx]
        neighbors_vis = vis_dyn[neighbors_idx]  # N x neighbors x F

        control_pts = control_pts.unsqueeze(1)  # N x 1 x F x 3

        diff = control_pts - neighbors_loc

        neighbors_norm = torch.norm(diff, dim=-1, p=2)

        neighbors_norm_diff = torch.abs(neighbors_norm[:,:,1:] - neighbors_norm[:,:,:-1]) # N x neighbors x F-1

        vis_mask_neigh = torch.logical_and(neighbors_vis[:,:,1:], neighbors_vis[:,:,:-1])
        vis_mask_curr = torch.logical_and(vis_dyn[:,1:], vis_dyn[:, :-1])
        vis_mask = torch.logical_and(vis_mask_curr.unsqueeze(1), vis_mask_neigh)
        
        # Debug information
        if debug:
            mask_sum = torch.sum(vis_mask).item()
            mask_size = vis_mask.numel()
            print(f"[laplacian_reg] Mask size: {mask_size}, True count: {mask_sum} ({mask_sum/mask_size*100:.2f}%)")
            
            if mask_sum > 0:
                valid_norm_diff = neighbors_norm_diff[vis_mask]
                print(f"[laplacian_reg] Valid distance difference stats - Min: {torch.min(valid_norm_diff).item():.6f}, Max: {torch.max(valid_norm_diff).item():.6f}, Mean: {torch.mean(valid_norm_diff).item():.6f}")
            else:
                print("[laplacian_reg] No valid neighbor pairs for calculating Laplacian loss")

        if outliers:
            outlier = torch.sum(neighbors_norm_diff * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-12)
            return outlier  # return outliers instead
        
        # Handling case where mask is all False
        if torch.sum(vis_mask) == 0:
            if debug:
                print("[laplacian_reg] Warning: Mask contains no True values, returning zero loss")
            return torch.tensor(0.0, device=self.device)

        laplacian_loss = torch.mean(neighbors_norm_diff[vis_mask])

        return laplacian_loss

    @torch.no_grad()
    def filter_dyn(self):

        if self.num_points_dyn == 0:
            return

        with torch.no_grad():
            self.get_neighbors()
            _, _, outliers_smooth = self.smooth_reg(filter_outlier=True)
            outliers_laplacian = self.laplacian_reg(outliers=True)
            outlier_score = self.opt.dyn_smooth_weight_t * outliers_smooth +  self.opt.dyn_laplacian_weight_t * outliers_laplacian
            outliers = torch.concat((outlier_score , outlier_score [:,-1:]), dim=-1)
            points_energy = torch.max(outliers, dim=1)[0]

            thrs = torch.quantile(points_energy, 0.98) + 1.0
            outliers = points_energy > thrs
            self.outliers_dyn = outliers

            self.all_vis_dyn[:, self.outliers_dyn] = False

    def optimize_dyn(self):

        """ 
        Optimization
        Loss Implemented:
        L_flow -> 2D reprojection loss of 3D points
        L_depth -> depth loss compared to pred depth
        """
        # Add call to visualize dynamic regions
        self.visualize_BA_regions()
        
        self.reset_optimizer()

        self.get_neighbors()

        logging.info("Starting stage 3 dynamic optimization")

        self.active_optimizers = ["points_dyn_t"]

        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)

        earlystopper = EarlyStopper(patience=100)

        if len(self.active_optimizers) != 0:

            for epoch in range(self.opt.num_dyn_epochs):

                for o in self.active_optimizers:
                    self.optimizers[o].zero_grad()
                
                loss_log = {}

                loss_smooth_t, _, _ = self.smooth_reg()
                loss_laplacian = self.laplacian_reg()

                total_reproj_loss = self.BA_fast(is_dyn=True)
                total_reproj_loss *= self.opt.reproj_weight

                total_loss =  total_reproj_loss + self.opt.dyn_smooth_weight_t * loss_smooth_t + self.opt.dyn_laplacian_weight_t * loss_laplacian
                
                loss_log["reproj_loss"] = total_reproj_loss.item()
                loss_log["smooth_t"] = self.opt.dyn_smooth_weight_t * loss_smooth_t.item()
                loss_log["laplacian_loss"] = self.opt.dyn_laplacian_weight_t * loss_laplacian.item()

                total_loss.backward()

                if earlystopper.early_stop(total_loss):
                    break

                for o in self.active_optimizers:
                    self.optimizers[o].step()
                    if o in self.schedulers:
                        self.schedulers[o].step(total_loss)

                if epoch % self.opt.print_every == 0:
                    logging.info(f"Finished epoch {epoch} with {loss_log}")
                    logging.info(f"Total loss: {total_loss.item()} with lr: {self.schedulers['points_dyn_t'].get_last_lr()}")

        logging.info("Finished optimization")

    def unfreeze_frames(self):

        for f in range(self.num_frames):
            pos_param = getattr(self.poses, f'delta_rotation_{f}')
            pos_param.requires_grad = True
            trans_param = getattr(
            self.poses, f'delta_translation_{f}')
            trans_param.requires_grad = True

    def ordered_ratio(self, disp_a, disp_b):
        ratio_a = torch.maximum(disp_a, disp_b) / \
            (torch.minimum(disp_a, disp_b)+1e-5)
        return ratio_a - 1
    
    def optimize_init(self):

        """
        Performs first stage of sliding window optimization to obtain a good initial pose estimate
        Assumes depth is GT for this stage. Performs pair wise warping
        """
        # Add call to visualize initialization regions
        self.visualize_init_regions()

        logging.info("Starting init optimization")

        self.active_optimizers = ["rotation_pose", "translation_pose", "intrinsics"]

        if not self.opt.opt_intrinsics:
            self.active_optimizers.remove("intrinsics")
        else:
            logging.info("Initial intrinsic matrix is: ")
            logging.info(self.get_intrinsics_K())

        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)


        self.keyframe_buffer = KeyFrameBuffer(buffer_size=5)
        self.keyframe_buffer.add_keyframe(0)
        self.keyframe_buffer.add_keyframe(1)

        self.optimize_over_keyframe_buffer()
        self.freeze_frame(self.keyframe_buffer.buffer[0])

        for x in range(2, self.num_frames):
            self.keyframe_buffer.add_keyframe(x)

            self.init_pose(x)  # just use the previous frame as init
            self.optimize_over_keyframe_buffer()
            self.freeze_frame(self.keyframe_buffer.buffer[0])
            
        self.unfreeze_frames()
        
        logging.info(f"Done with init optimization")

    def optimize_over_pairs(self, pairs):
        all_pairs = torch.tensor(pairs)
        
        Rs_from, ts_from = self.get_poses(all_pairs[:, 0])         # N x 3 x 3,     N x 3 x 1
        Rs_to, ts_to = self.get_poses(all_pairs[:, 1])

        K_from = self.get_intrinsics_K(all_pairs[:, 0])
        K_to = self.get_intrinsics_K(all_pairs[:, 1])

        static_tracks = self.all_tracks_static_init_filtered[all_pairs[:, 0]]
        B, N, _ = static_tracks.shape
        static_homo_from = torch.concatenate((static_tracks, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        
        static_cam_from = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_from), static_homo_from)
        static_cam_from =  static_cam_from / (static_cam_from[:,:,2].unsqueeze(-1)+1e-16)
        
        # Modification: Use filtered depth information
        if hasattr(self, 'all_tracks_static_depth_init_filtered') and self.all_tracks_static_depth_init_filtered is not None:
            static_cam_from = static_cam_from * self.all_tracks_static_depth_init_filtered[all_pairs[:,0]].unsqueeze(-1)
        else:
            # If filtered depth information is unavailable, extract corresponding depth from original depth
            # This requires knowing valid_static_indices, but we might not have it here, so it's best to ensure depth information is correctly created in create_init_static_points
            logging.warning("Using original depth information - this may cause dimension mismatch")
            static_cam_from = static_cam_from * self.all_tracks_static_depth[all_pairs[:,0]].unsqueeze(-1)

        static_world_from = torch.einsum("bni,bmi->bmn", Rs_from, static_cam_from) + ts_from.permute(0,2,1)    # F x N x 3

        static_cam_to = torch.einsum("bin,bmi->bmn", Rs_to, static_world_from - ts_to.permute(0,2,1))         # F x N x 3

        static_coord = torch.einsum("bji,bni->bnj", K_to, static_cam_to)
        static_coord = static_coord / (static_coord[:,:,2].unsqueeze(-1)+1e-16)
        static_coord = static_coord[:,:,:2]

        # Modification: Use filtered visibility information
        static_tracks_vis_from = self.all_vis_static_init_filtered[all_pairs[:, 0]]
        static_tracks_vis_to = self.all_vis_static_init_filtered[all_pairs[:, 1]]
        mask = torch.logical_and(static_tracks_vis_from, static_tracks_vis_to).float()

        static_tracks_to = self.all_tracks_static_init_filtered[all_pairs[:, 1]]

        # flow_error = self.loss_fn(gt_coord - pred_coord, self.flow_norm_l1)
        flow_error = flow_norm(static_tracks_to - static_coord, self.flow_norm_l1)
        flow_error = torch.sum(flow_error * mask, dim=1) / (torch.sum(mask, dim=1)+1e-16)
        flow_error = torch.mean(flow_error)

        return flow_error
    
    def optimize_over_keyframe_buffer(self):

        """
        Optimize using 2D pair wise flow over exhaustive pair wise matching in keyframe buffers
        """

        all_pairs = list(itertools.permutations(self.keyframe_buffer.buffer, 2))
        
        # all_pairs = list(itertools.combinations(self.keyframe_buffer.buffer, 2))

        earlystopper = EarlyStopper(patience=20)

        for epoch in range(self.opt.num_init_epochs):

            for o in self.active_optimizers:
                self.optimizers[o].zero_grad()

            total_reproj_loss = self.optimize_over_pairs(all_pairs)

            total = total_reproj_loss * self.opt.reproj_weight

            if self.opt.pose_smooth_weight_t > 0 and len(self.keyframe_buffer.buffer) >= 3:
                loss_pose_smooth_t, loss_poss_smoother_R = self.calc_pose_smooth_loss(self.keyframe_buffer.buffer)
                loss_pose_smooth_t = loss_pose_smooth_t * self.opt.pose_smooth_weight_t
                loss_pose_smooth_R = loss_poss_smoother_R * self.opt.pose_smooth_weight_r
            else:
                loss_pose_smooth_t = torch.tensor(0)
                loss_pose_smooth_R = torch.tensor(0)

            total += (loss_pose_smooth_t + loss_pose_smooth_R)
            total.backward()

            if earlystopper.early_stop(total):
                break

            for o in self.active_optimizers:
                self.optimizers[o].step()
                if o in self.schedulers:
                    self.schedulers[o].step(total)

            if epoch % self.opt.print_every == 0:
                logging.info(f"Currently at epoch {epoch} out of {self.opt.num_init_epochs} for frames {self.keyframe_buffer.buffer[0]}-{self.keyframe_buffer.buffer[-1]} with pose smooth loss t: {loss_pose_smooth_t.item()} * {self.opt.pose_smooth_weight_t}, pose smooth loss R: {loss_pose_smooth_R.item()} * {self.opt.pose_smooth_weight_r}, reproj loss: {total_reproj_loss.item()} * {self.opt.reproj_weight}")
                logging.info(f"Total loss: {total.item()} with lr: {self.schedulers['translation_pose'].get_last_lr()}")

        self.reset_optimizer()

    def check_flow_compatibility(self):
        """
        Check compatibility between current optical flow model and image dimensions
        
        Returns:
            bool: True if compatible, False otherwise
        """
        if not hasattr(self, 'images') or len(self.images) < 2:
            logging.warning("Images not yet loaded, cannot check optical flow compatibility")
            return False
            
        try:
            # Test optical flow prediction, check for size mismatch
            test_flow = self.get_flow_prediction(0, 1)
            flow_h, flow_w = test_flow.shape[:2]
            expected_h, expected_w = self.shape
            
            if flow_h == expected_h and flow_w == expected_w:
                logging.info(f"✅ Optical flow compatibility check passed: {flow_h}x{flow_w}")
                return True
            else:
                logging.warning(f"⚠️ Optical flow size mismatch: Flow {flow_h}x{flow_w} vs Expected {expected_h}x{expected_w}")
                return False
                
        except ValueError as e:
            if "Optical flow size mismatch" in str(e): # Assuming the error message might be translated or original
                logging.error("❌ Optical flow compatibility check failed: Size mismatch and interpolation forbidden")
                return False
            else:
                logging.error(f"❌ Optical flow compatibility check failed: {e}")
                return False
        except Exception as e:
            logging.error(f"❌ Exception during optical flow compatibility check: {e}")
            return False

    def auto_adjust_resolution_for_flow(self):
        """
        Automatically adjust system resolution to match optical flow model output, avoiding interpolation
        
        Warning: This will modify self.shape, possibly affecting other components
        """
        if not hasattr(self, 'images') or len(self.images) < 2:
            logging.error("Images not yet loaded, cannot adjust resolution")
            return False
            
        # Temporarily enable interpolation to get optical flow output size
        original_setting = getattr(self.opt, 'allow_flow_interpolation', False)
        self.opt.allow_flow_interpolation = True
        
        try:
            test_flow = self.get_flow_prediction(0, 1)
            flow_h, flow_w = test_flow.shape[:2]
            original_h, original_w = self.shape
            
            if flow_h == original_h and flow_w == original_w:
                logging.info(f"✅ Resolution matches, no adjustment needed: {flow_h}x{flow_w}")
                return True
                
            # Adjust system resolution
            self.shape = (flow_h, flow_w)
            self.height = flow_h
            self.width = flow_w
            
            # Update resolution registered in intrinsics
            if hasattr(self, 'intrinsics'):
                self.intrinsics.register_shape(self.shape)
            
            logging.info(f"🔄 Automatically adjusted system resolution: {original_h}x{original_w} -> {flow_h}x{flow_w}")
            logging.warning("⚠️ Resolution adjustment may affect compatibility of depth maps, masks, etc.")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Automatic resolution adjustment failed: {e}")
            return False
        finally:
            # Restore original interpolation setting
            self.opt.allow_flow_interpolation = original_setting

    def save_state(self, state_file=None):
        """
        Save current engine state, containing all key data to ensure consistency after loading
        """
        if state_file is None:
            state_file = os.path.join(self.opt.output_dir, "engine_state.pth")
        
        # Record key metrics of state before saving
        logging.info(f"Diagnostics before saving engine state:")
        if hasattr(self, 'depth') and self.depth is not None:
            logging.info(f"- self.depth: shape={self.depth.shape}, Stats={self.depth.mean().item():.4f}±{self.depth.std().item():.4f}")
        if hasattr(self, 'reproj_depths') and self.reproj_depths is not None:
            logging.info(f"- self.reproj_depths: shape={self.reproj_depths.shape}, Stats={self.reproj_depths.mean().item():.4f}±{self.reproj_depths.std().item():.4f}")
        
        # Record mask information
        if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
            logging.info(f"- Saving dyn_masks_filters_deva: shape={self.dyn_masks_filters_deva.shape}")
            dynamic_pixels = np.sum(self.dyn_masks_filters_deva == 1)
            total_pixels = self.dyn_masks_filters_deva.size
            logging.info(f"  DeVA dynamic region proportion: {dynamic_pixels}/{total_pixels} ({dynamic_pixels/total_pixels*100:.2f}%)")
        else:
            logging.warning("- Warning: dyn_masks_filters_deva does not exist, will not save this attribute")
        
        # Record camera pose information
        if hasattr(self, 'poses'):
            logging.info("- Camera poses before saving:")
            try:
                for i in range(min(5, self.num_frames)):
                    Rs, ts = self.get_poses([i])
                    logging.info(f"  Frame {i} Pose: R det={torch.linalg.det(Rs[0]).item():.4f}, R trace={torch.trace(Rs[0]).item():.4f}, t={ts[0].squeeze().detach().cpu().numpy()}")
            except Exception as e:
                logging.error(f"Error recording camera poses: {e}")
        
        # Collect state to be saved
        state_dict = {
            # Save control points
            'controlpoints_static': self.controlpoints_static.state_dict(),
            'controlpoints_dyn': self.controlpoints_dyn.state_dict() if hasattr(self, 'controlpoints_dyn') else None,
            # Save camera poses
            'poses': self.poses.state_dict(),
            # Save feature points and depth information
            'all_tracks_static': self.all_tracks_static,
            'all_vis_static': self.all_vis_static,
            'all_tracks_dyn': self.all_tracks_dyn,
            'all_vis_dyn': self.all_vis_dyn,
            'all_tracks_static_depth': self.all_tracks_static_depth,
            'all_tracks_dyn_depth': self.all_tracks_dyn_depth if hasattr(self, 'all_tracks_dyn_depth') else None,
            # Save camera intrinsics
            'K_gt': self.K_gt if hasattr(self, 'K_gt') else None,
            'K': self.K if hasattr(self, 'K') else None,
            'intrinsics': self.intrinsics.state_dict() if hasattr(self, 'intrinsics') and self.opt.opt_intrinsics else None,
            # Save other important parameters
            'dyn_masks': self.dyn_masks,
            'dyn_masks_filters': self.dyn_masks_filters,
            'dyn_masks_filters_deva': self.dyn_masks_filters_deva if hasattr(self, 'dyn_masks_filters_deva') else None,
            'num_points_static': self.num_points_static,
            'num_points_dyn': self.num_points_dyn,
            'num_frames': self.num_frames,
            'shape': self.shape,
            # [New] Save original depth and interpolated depth
            'depth_parameter': self.depth.detach() if hasattr(self, 'depth') and isinstance(self.depth, torch.nn.Parameter) else None,
            'reproj_depths': self.reproj_depths.detach() if hasattr(self, 'reproj_depths') and self.reproj_depths is not None else None,
            # [New] Record depth interpolation state
            'depth_interpolated': hasattr(self, 'reproj_depths') and self.reproj_depths is not None,
        }
        
        # Save state to file
        # torch.save(state_dict, state_file)
        # logging.info(f"Engine state saved to: {state_file}")
    
    def load_state(self, state_file=None):
        """
        Load previously saved engine state to ensure optical flow optimization consistency
        """
        if state_file is None:
            state_file = os.path.join(self.opt.output_dir, "engine_state.pth")
        
        print(f"Attempting to load state file: {state_file}")
        
        if not os.path.exists(state_file):
            print(f"State file does not exist: {state_file}")
            logging.error(f"State file does not exist: {state_file}")
            return False
        
        try:
            # Load state
            print(f"Processing loading engine state: {state_file}")
            logging.info(f"Processing loading engine state: {state_file}")
            state_dict = torch.load(state_file, map_location=self.device)
            
            # Clear cache because data may have changed
            self._cached_cotracker_mask = None
            
            # Record key information of loaded state
            if 'depth_parameter' in state_dict and state_dict['depth_parameter'] is not None:
                depth_tensor = state_dict['depth_parameter']
                logging.info(f"Depth map in state file: shape={depth_tensor.shape}, Stats={depth_tensor.mean().item():.4f}±{depth_tensor.std().item():.4f}")
            
            if 'reproj_depths' in state_dict and state_dict['reproj_depths'] is not None:
                reproj_depths = state_dict['reproj_depths']
                logging.info(f"Reprojected depths in state file: shape={reproj_depths.shape}, Stats={reproj_depths.mean().item():.4f}±{reproj_depths.std().item():.4f}")
            
            # --- 1. Restore basic information and data not dependent on other modules ---
            self.num_points_static = state_dict['num_points_static']
            self.num_points_dyn = state_dict['num_points_dyn']
            self.num_frames = state_dict['num_frames']
            self.shape = state_dict['shape']
            print(f"Restored points from state file: static={self.num_points_static}, dyn={self.num_points_dyn}, frames={self.num_frames}, shape={self.shape}")

            self.all_tracks_static = state_dict['all_tracks_static'].to(self.device)
            self.all_vis_static = state_dict['all_vis_static'].to(self.device)
            self.all_tracks_dyn = state_dict['all_tracks_dyn'].to(self.device)
            self.all_vis_dyn = state_dict['all_vis_dyn'].to(self.device)
            self.all_tracks_static_depth = state_dict['all_tracks_static_depth'].to(self.device)
            if state_dict.get('all_tracks_dyn_depth') is not None:
                self.all_tracks_dyn_depth = state_dict['all_tracks_dyn_depth'].to(self.device)
            
            # Ensure masks are also on the correct device
            self.dyn_masks = state_dict['dyn_masks'].to(self.device)
            # dyn_masks_filters is a numpy array, no need to move to device
            self.dyn_masks_filters = state_dict['dyn_masks_filters']

            # Check if dyn_masks_filters_deva exists, create if not
            if not 'dyn_masks_filters_deva' in state_dict or state_dict['dyn_masks_filters_deva'] is None:
                logging.warning("dyn_masks_filters_deva missing in loaded state, attempting to regenerate from dyn_masks_filters")
                try:
                    # Get shape of all mask arrays
                    h, w = self.shape
                    num_frames = self.dyn_masks_filters.shape[0]
                    
                    # Create dyn_masks_filters_deva array
                    dyn_masks_filters_deva_list = []
                    
                    # Generate masks removing EPI influence based on existing dyn_masks_filters
                    for f in range(num_frames):
                        mask_dyn_pro = np.copy(self.dyn_masks_filters[f])
                        
                        # Change label 3 (EPI dynamic) to other labels
                        # This is a heuristic method:
                        # If pixel is EPI dynamic (label 3), we assume it might be DeVA static (label 0)
                        # Because DeVA static being covered by EPI is the most common case
                        mask_dyn_deva_only = np.copy(mask_dyn_pro)
                        epi_mask = (mask_dyn_pro == 3)
                        mask_dyn_deva_only[epi_mask] = 0  # Reset EPI dynamic regions to static regions
                        
                        dyn_masks_filters_deva_list.append(mask_dyn_deva_only)
                    
                    # Convert to numpy array and save
                    self.dyn_masks_filters_deva = np.array(dyn_masks_filters_deva_list).astype(np.uint8)
                    logging.info("Successfully regenerated dyn_masks_filters_deva from dyn_masks_filters")
                    
                    # Output statistics
                    total_deva_dynamic = np.sum(self.dyn_masks_filters_deva == 1)
                    total_epi_converted = np.sum(epi_mask)
                    total_pixels = self.dyn_masks_filters_deva.size
                    logging.info(f"DeVA dynamic region proportion in regenerated dyn_masks_filters_deva: {total_deva_dynamic/total_pixels*100:.2f}%")
                    logging.info(f"Number of pixels converted from EPI dynamic labels: {total_epi_converted}")
                except Exception as e:
                    logging.error(f"Error regenerating dyn_masks_filters_deva: {e}")
                    logging.warning("Option exclude_epi_from_flow=True will be unavailable")
            else:
                self.dyn_masks_filters_deva = state_dict['dyn_masks_filters_deva']
                logging.info("Successfully loaded dyn_masks_filters_deva")
            
            # --- 2. Restore camera intrinsics ---
            self.K = None 
            if 'K' in state_dict and state_dict['K'] is not None:
                 self.K = state_dict['K'].to(self.device)
                 print(f"Loaded self.K, shape: {self.K.shape}")
            
            self.K_gt = None
            if 'K_gt' in state_dict and state_dict['K_gt'] is not None:
                self.K_gt = state_dict['K_gt'].to(self.device)
                print(f"Loaded self.K_gt, shape: {self.K_gt.shape}")
            
            if not self.opt.opt_intrinsics and self.K_gt is None and self.K is not None:
                self.K_gt = self.K
                logging.warning("K_gt missing in state file, using K as K_gt")
                print("Warning: K_gt missing in state file, using K as K_gt")

            # --- 3. Restore depth data ---
            # Restore depth map parameters from saved state
            if 'depth_parameter' in state_dict and state_dict['depth_parameter'] is not None:
                print("Restoring depth map parameters from state file")
                depth_tensor = state_dict['depth_parameter'].to(self.device)
                self.depth = torch.nn.Parameter(depth_tensor, requires_grad=True)
                logging.info(f"Restored depth map parameters from state file, shape: {self.depth.shape}")
            
            # Restore reproj depths from saved state
            if 'reproj_depths' in state_dict and state_dict['reproj_depths'] is not None:
                print("Restoring reprojection depth map from state file")
                self.reproj_depths = state_dict['reproj_depths'].to(self.device)
                logging.info(f"Restored reprojection depth map from state file, shape: {self.reproj_depths.shape}")
                
                # Simultaneously create optimizable parameter version for subsequent optimization
                self.reproj_depths_param = torch.nn.Parameter(self.reproj_depths.clone(), requires_grad=True)
                logging.info("Created optimizable parameter reproj_depths_param from reprojected depths")

            # --- 3. Create/Restore CameraIntrinsics ---
            if self.opt.opt_intrinsics:
                # Initialize intrinsics 
                initial_fx, initial_fy = 1.0, 1.0 
                k_source_found = False
                
                # Determine initial values by priority
                if self.K is not None:
                    try:
                        k_init_matrix = self.K[0] if len(self.K.shape) == 3 else self.K
                        initial_fx = k_init_matrix[0, 0].item()
                        initial_fy = k_init_matrix[1, 1].item()
                        k_source_found = True
                        print(f"Using loaded K to initialize intrinsics: fx={initial_fx}, fy={initial_fy}")
                    except (IndexError, TypeError) as e:
                        logging.warning(f"Loaded K format unexpected ({e}), cannot extract fx, fy")
                
                elif self.K_gt is not None and not k_source_found:
                    try:
                        initial_fx = self.K_gt[0, 0, 0].item() 
                        initial_fy = self.K_gt[0, 1, 1].item()
                        k_source_found = True
                        print(f"Using loaded K_gt to initialize intrinsics: fx={initial_fx}, fy={initial_fy}")
                    except (IndexError, TypeError) as e:
                        logging.warning(f"Loaded K_gt format unexpected ({e}), cannot extract fx, fy")
                
                # Create instance
                self.intrinsics = CameraIntrinsics(initial_fx, initial_fy).to(self.device)
                self.intrinsics.register_shape(self.shape)
                print(f"Created CameraIntrinsics instance (fx={(initial_fx if k_source_found else 1.0)}, fy={(initial_fy if k_source_found else 1.0)})")

                # Load intrinsics state dict
                if 'intrinsics' in state_dict and state_dict['intrinsics'] is not None:
                    print("Loading CameraIntrinsics state_dict...")
                    self.intrinsics.load_state_dict(state_dict['intrinsics'])
                    print("CameraIntrinsics state_dict loaded.")
            
            # --- 4. Re-initialize control points module based on restored point counts ---
            self.controlpoints_static = ControlPoints(number_of_points=self.num_points_static).to(self.device)
            self.controlpoints_dyn = ControlPointsDynamic(
                number_of_points=self.num_points_dyn, 
                number_of_frames=self.num_frames, 
                with_norm=False
            ).to(self.device)
            
            # --- 5. Load state dicts for control points and poses ---
            self.controlpoints_static.load_state_dict(state_dict['controlpoints_static'])
            if state_dict['controlpoints_dyn'] is not None:
                self.controlpoints_dyn.load_state_dict(state_dict['controlpoints_dyn'])
            
            # Record statistics before loading camera poses
            logging.info("Statistics before loading camera poses:")
            try:
                if hasattr(self, 'poses'):
                    # Extract poses of some keyframes
                    for i in range(min(3, self.num_frames)):
                        Rs, ts = self.get_poses([i])
                        logging.info(f"- Frame {i} Pose: R det={torch.linalg.det(Rs[0]).item():.4f}, R trace={torch.trace(Rs[0]).item():.4f}, t={ts[0].squeeze().detach().cpu().numpy()}")
            except Exception as e:
                logging.error(f"Error recording statistics before loading poses: {e}")
            
            # Load camera poses
            self.poses = CameraPoseDeltaCollection(self.num_frames).to(self.device)
            self.poses.load_state_dict(state_dict['poses'])
            
            # Record statistics after loading camera poses
            logging.info("Statistics after loading camera poses:")
            try:
                # Extract poses of some keyframes
                for i in range(min(3, self.num_frames)):
                    Rs, ts = self.get_poses([i])
                    logging.info(f"- Frame {i} Pose: R det={torch.linalg.det(Rs[0]).item():.4f}, R trace={torch.trace(Rs[0]).item():.4f}, t={ts[0].squeeze().detach().cpu().numpy()}")
            except Exception as e:
                logging.error(f"Error recording statistics after loading poses: {e}")
            
            # --- 6. Create is_static_strict_tensor ---
            if hasattr(self, 'dyn_masks_filters') and self.dyn_masks_filters is not None:
                h, w = self.shape
                num_f_effective = self.dyn_masks_filters.shape[0]
                self.is_static_strict = np.zeros((num_f_effective, h, w), dtype=bool)
                for f_idx in range(num_f_effective):
                    # Modification: Only regions with label 0 are strictly static regions
                    # Label 1=DeVA dynamic, Label 2=Boundary/rejected region, Label 3=EPI dynamic
                    self.is_static_strict[f_idx] = (self.dyn_masks_filters[f_idx] == 0)
                    
                    # Count pixels for each label for debugging
                    label_counts = np.bincount(self.dyn_masks_filters[f_idx].flatten(), minlength=4)
                    logging.debug(f"Frame {f_idx} label statistics: Static(0)={label_counts[0]}, DeVA Dynamic(1)={label_counts[1]}, Boundary/Rejected(2)={label_counts[2]}, EPI Dynamic(3)={label_counts[3]}")

                self.is_static_strict_tensor = torch.from_numpy(self.is_static_strict).bool().to(self.device)
                logging.info(f"Created/Updated is_static_strict_tensor based on dyn_masks_filters == 0, shape: {self.is_static_strict_tensor.shape}")
                
                # Add more detailed log output showing the proportion of static regions
                total_pixels = self.is_static_strict.size
                static_pixels = np.sum(self.is_static_strict)
                logging.info(f"Strict static region proportion: {static_pixels}/{total_pixels} ({static_pixels/total_pixels*100:.2f}%)")
            elif hasattr(self, 'is_static_strict_tensor'): 
                del self.is_static_strict_tensor

            print(f"Engine state loaded successfully")
            logging.info(f"Engine state loaded successfully")
            
            # Force recalculation of reproj_depths after loading basic parameters
            if 'depth_parameter' in state_dict and state_dict['depth_parameter'] is not None:
                # Load original depth parameters but not reproj_depths
                self.depth = torch.nn.Parameter(state_dict['depth_parameter'].to(self.device), requires_grad=True)
                
                # Re-execute depth interpolation instead of loading saved interpolation results
                if not ('reproj_depths' in state_dict and state_dict['reproj_depths'] is not None):
                    self.depth_interpolate(make_optimizable=True)
                    logging.info("Recalculated initial interpolated depths, ignoring saved interpolation results")
                
            # Regenerate Co-tracker static point mask cache (since data has updated)
            logging.info("Regenerating Co-tracker static point mask cache...")
            self.get_cotracker_static_mask()
            logging.info("Co-tracker static point mask cache regenerated.")
            
            return True
        except Exception as e:
            print(f"Error loading engine state: {e}")
            logging.error(f"Error loading engine state: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def optimize_with_dense_flow(self):
        """
        Optimize depth and pose for all static regions using dense optical flow model
        Does not depend on cotracker control points, directly optimizes the entire static region
        """

        
        logging.info("Starting static region optimization based on dense optical flow")
        
        # Add diagnostic function
        self.check_flow_optimization_parameters()
        
        # Save depth map before optimization
        depth_before = None
        if hasattr(self, 'reproj_depths_param') and self.reproj_depths_param is not None:
            depth_before = self.reproj_depths_param.detach().clone()
            logging.info("Saved depth map before optimization for subsequent analysis")

        # Check and report optical flow optimization mode
        use_weighted_flow = getattr(self.opt, 'use_weighted_flow', True)
        if use_weighted_flow:
            logging.info("Using weighted optical flow optimization mode (use_weighted_flow=True)")
            logging.info("  - Will use forward-backward flow consistency check")
            logging.info("  - Will apply consistency-based Gaussian weights")
            logging.info("  - Supports advanced features like truncation threshold and minimum weight")
        else:
            logging.info("Using simple optical flow optimization mode (use_weighted_flow=False)")
            logging.info("  - Directly calculate flow loss, without consistency weights")
            logging.info("  - All pixels use equal weight")
            logging.info("  - Similar to simple flow calculation in backup code")

        # Set default parameter values
        if not hasattr(self.opt, 'flow_weight'):
            self.opt.flow_weight = 1.0
            logging.info(f"Set default flow_weight={self.opt.flow_weight}")

        if not hasattr(self.opt, 'flow_model'):
            self.opt.flow_model = "unimatch"
            logging.info(f"Set default flow_model={self.opt.flow_model}")

        if not hasattr(self.opt, 'num_flow_epochs'):
            self.opt.num_flow_epochs = 100
            logging.info(f"Set default num_flow_epochs={self.opt.num_flow_epochs}")
        
        if not hasattr(self.opt, 'flow_opt_pose'):
            self.opt.flow_opt_pose = False
            logging.info(f"Set default flow_opt_pose={self.opt.flow_opt_pose}")
            
        if not hasattr(self.opt, 'flow_opt_depth'):
            self.opt.flow_opt_depth = True
            logging.info(f"Set default flow_opt_depth={self.opt.flow_opt_depth}")
            
        if not hasattr(self.opt, 'flow_opt_intrinsics'):
            self.opt.flow_opt_intrinsics = self.opt.opt_intrinsics
            logging.info(f"Set default flow_opt_intrinsics={self.opt.flow_opt_intrinsics}")
            
        # Check if forced regeneration of dyn_masks_filters_deva is needed
        if hasattr(self.opt, 'force_regenerate_deva_masks') and self.opt.force_regenerate_deva_masks:
            logging.info("Detected force_regenerate_deva_masks=True, will force regenerate dyn_masks_filters_deva")
            success = self.regenerate_deva_masks()
            if success:
                logging.info("Successfully forced regeneration of dyn_masks_filters_deva")
            else:
                logging.warning("Forced regeneration of dyn_masks_filters_deva failed, optical flow optimization may be affected")
            
        # Handle EPI exclusion parameters
        if hasattr(self.opt, 'exclude_epi_from_flow') and self.opt.exclude_epi_from_flow:
            logging.info("Detected exclude_epi_from_flow=True parameter")
            logging.info("Will use the negation of pure DeVA dynamic regions unaffected by EPI as optical flow optimization regions")
            logging.info("This ensures optical flow optimization is only performed in non-dynamic regions identified by DeVA, completely ignoring EPI regions")
        else:
            logging.info("exclude_epi_from_flow not set or False")
            logging.info("Will use DeVA static regions (label 0) as optical flow optimization regions")
        
        # Update static mask used for optical flow optimization
        self.update_static_mask_for_flow()
        
        # Visualize regions used for optical flow optimization
        self.visualize_flow_optimization_regions()
        
        # Ensure reproj_depths_param is created and optimizable
        if self.opt.flow_opt_depth:
            if not hasattr(self, 'reproj_depths_param') or self.reproj_depths_param is None:
                # First ensure reproj_depths exists
                if not hasattr(self, 'reproj_depths') or self.reproj_depths is None:
                    logging.warning("reproj_depths does not exist, attempting to create via depth_interpolate")
                    try:
                        self.depth_interpolate(make_optimizable=True)
                    except Exception as e:
                        logging.error(f"depth_interpolate Failed: {e}")
                        logging.info("Attempting to create basic depth map from initial depth")
                        # Create a basic depth map
                        if hasattr(self, 'shape'):
                            initial_depth = torch.ones(self.num_frames, *self.shape, device=self.device) * 5.0
                            self.reproj_depths = initial_depth
                            self.reproj_depths_param = torch.nn.Parameter(initial_depth.clone(), requires_grad=True)
                        else:
                            logging.error("Cannot create depth parameter, shape information missing")
                            return
                else:
                    # Create optimizable parameter from existing reproj_depths
                    logging.info("Creating optimizable parameter from existing reproj_depths")
                    self.reproj_depths_param = torch.nn.Parameter(self.reproj_depths.clone(), requires_grad=True)
            
            # Verify validity of depth parameters
            if hasattr(self, 'reproj_depths_param') and self.reproj_depths_param is not None:
                if torch.isnan(self.reproj_depths_param).any() or torch.isinf(self.reproj_depths_param).any():
                    logging.error("Depth parameters contain invalid values, fixing")
                    # Fix invalid values
                    valid_mask = torch.isfinite(self.reproj_depths_param)
                    self.reproj_depths_param.data[~valid_mask] = 5.0  # Set default depth value
            
            # Ensure reproj_depth optimizer exists
            if 'reproj_depth' not in self.optimizers and hasattr(self, 'reproj_depths_param'):
                reproj_depth_lr = getattr(self.opt, 'reproj_depth_lr', getattr(self.opt, 'depth_lr', 1e-4))
                self.reproj_depth_optimizer = torch.optim.Adam([self.reproj_depths_param], lr=reproj_depth_lr, weight_decay=0, amsgrad=True)
                self.optimizers["reproj_depth"] = self.reproj_depth_optimizer
                logging.info(f"Created reproj_depth optimizer, lr={reproj_depth_lr}")

        # Sliding window size
        window_size = min(5, self.num_frames)
        if window_size < 2: # At least 2 frames are needed to form a pair
            logging.warning(f"Video frame count ({self.num_frames}) too low, cannot perform optical flow optimization.")
            return

        try:
            logging.info("Starting loading optical flow model...")
            self.load_flow_model()
            
            # Check if optical flow processor loaded successfully
            if hasattr(self, 'flow_processor') and self.flow_processor is not None:
                logging.info(f"✅ Optical flow processor loaded successfully: {type(self.flow_processor)}")
            else:
                logging.error("❌ Optical flow processor load failed: flow_processor is None")
                return

            # --- Determine objects to optimize based on ablation study parameters ---
            self.active_optimizers = []
            opt_components = [] 
            if self.opt.flow_opt_depth:
                if "reproj_depth" in self.optimizers:
                    self.active_optimizers.append("reproj_depth")
                    opt_components.append("ReprojDepth")
                else:
                    logging.warning("Requested flow_opt_depth=True but reproj_depth optimizer not found.")
            if self.opt.flow_opt_pose:
                if "rotation_pose" in self.optimizers and "translation_pose" in self.optimizers:
                    self.active_optimizers.extend(["rotation_pose", "translation_pose"])
                    opt_components.append("Pose")
                else:
                    logging.warning("Requested flow_opt_pose=True but pose optimizers not found.")
            if self.opt.flow_opt_intrinsics:
                if self.opt.opt_intrinsics:
                    if "intrinsics" in self.optimizers:
                        self.active_optimizers.append("intrinsics")
                        opt_components.append("Intrinsics")
                    else:
                        logging.warning("Requested flow_opt_intrinsics=True but intrinsics optimizer not found.")
                else:
                     logging.warning("Requested flow_opt_intrinsics=True but global opt_intrinsics is False.")

            if not self.active_optimizers:
                logging.warning("No components selected for optimization in dense flow phase. Skipping.")
                return

            logging.info(f"Dense flow optimization active components: {' + '.join(opt_components) if opt_components else 'None'}")
            logging.info(f"Dense flow optimization active optimizers: {self.active_optimizers}")
            
            # --- Precompute optical flow for all frame pairs in all possible windows ---
            logging.info("Precomputing all required optical flows...")
            logging.info(f"Total video frames: {self.num_frames}, Window size: {window_size}")
            
            all_required_pairs = set()
            for start_idx_scan in range(0, self.num_frames - 1, window_size - 1 if window_size > 1 else 1):
                end_idx_scan = min(start_idx_scan + window_size, self.num_frames)
                if end_idx_scan - start_idx_scan < 2: continue # Skip windows that cannot form pairs
                frames_scan = list(range(start_idx_scan, end_idx_scan))
                all_pairs_scan = list(itertools.permutations(frames_scan, 2))
                logging.debug(f"Window {start_idx_scan}-{end_idx_scan-1}: Frames={frames_scan}, Generating {len(all_pairs_scan)} pairs")
                for pair in all_pairs_scan:
                    all_required_pairs.add(tuple(sorted(pair))) # Store normalized frame pairs to avoid duplicate calculation (0,1) and (1,0)
            
            logging.info(f"Total unique pairs to precompute: {len(all_required_pairs)}")

            precomputed_flow_predictions = {}
            # TQDM progress bar
            if hasattr(self.opt, 'debug_mode') and self.opt.debug_mode:
                pair_iterator = tqdm(list(all_required_pairs), desc="Precomputing Flows")
            else:
                pair_iterator = list(all_required_pairs)

            success_count = 0
            fail_count = 0
            
            for from_idx_p, to_idx_p in pair_iterator:
                if (from_idx_p, to_idx_p) not in precomputed_flow_predictions:
                     try:
                         # Get raw optical flow prediction
                         logging.debug(f"Starting to get flow prediction: {from_idx_p}->{to_idx_p}")
                         raw_flow = self.get_flow_prediction(from_idx_p, to_idx_p)
                         
                         if raw_flow is None:
                             logging.error(f"❌ Failed to get flow prediction: {from_idx_p}->{to_idx_p} (raw_flow is None)")
                             fail_count += 1
                             continue
                         
                         logging.debug(f"Got flow prediction, size: {raw_flow.shape}")
                         
                         # Skip optical flow consistency check, use raw optical flow prediction directly
                         precomputed_flow_predictions[(from_idx_p, to_idx_p)] = raw_flow
                         success_count += 1
                     except Exception as e:
                         logging.error(f"❌ Error precomputing flow {from_idx_p}->{to_idx_p}: {e}")
                         import traceback
                         logging.debug(traceback.format_exc())
                         fail_count += 1
                         
                if (to_idx_p, from_idx_p) not in precomputed_flow_predictions: # Calculate reverse flow as well
                     try:
                         # Get reverse flow prediction
                         logging.debug(f"Starting to get reverse flow prediction: {to_idx_p}->{from_idx_p}")
                         raw_flow_back = self.get_flow_prediction(to_idx_p, from_idx_p)
                         
                         if raw_flow_back is None:
                             logging.error(f"❌ Failed to get reverse flow prediction: {to_idx_p}->{from_idx_p} (raw_flow is None)")
                             fail_count += 1
                             continue
                         
                         # Skip reverse flow consistency check, use raw optical flow prediction directly
                         precomputed_flow_predictions[(to_idx_p, from_idx_p)] = raw_flow_back
                         logging.debug(f"✅ Precomputed reverse flow (Skip validation): {to_idx_p}->{from_idx_p}")
                         success_count += 1
                     except Exception as e:
                         logging.error(f"❌ Error precomputing reverse flow {to_idx_p}->{from_idx_p}: {e}")
                         import traceback
                         logging.debug(traceback.format_exc())
                         fail_count += 1
            
            logging.info(f"Optical flow precomputation completed: Success={success_count}, Failed={fail_count}, Total={len(precomputed_flow_predictions)} predictions")
            
            if len(precomputed_flow_predictions) == 0:
                logging.error("❌❌❌ No optical flows successfully precomputed! Optical flow optimization cannot proceed")
                return
            # --- Optical flow precomputation end ---

            # Use sliding window strategy
            for start_idx in range(0, self.num_frames - 1, window_size - 1 if window_size > 1 else 1):
                end_idx = min(start_idx + window_size, self.num_frames)
                if end_idx - start_idx < 2: continue # Skip windows that cannot form pairs

                logging.info(f"Optimizing dense flow for frame window: {start_idx}-{end_idx-1}")
                
                current_window_frames = list(range(start_idx, end_idx))
                current_window_pairs = list(itertools.permutations(current_window_frames, 2))
                
                # Get required flow from precomputed results for current window
                current_flow_predictions = {}
                logging.debug(f"Current window pairs: {len(current_window_pairs)}")
                
                found_count = 0
                missing_count = 0
                
                for p_from, p_to in current_window_pairs:
                    if (p_from, p_to) in precomputed_flow_predictions:
                        current_flow_predictions[f"{p_from}_{p_to}"] = precomputed_flow_predictions[(p_from, p_to)]
                        found_count += 1
                        logging.debug(f"✅ Found precomputed flow: {p_from}->{p_to}")
                    else:
                        missing_count += 1
                        # Theoretically shouldn't happen as we precomputed all sorted pairs forward/backward
                        logging.warning(f"❌ Flow pair ({p_from}, {p_to}) not found in precomputed, attempting recalculation")
                        
                        try:
                            # Recalculate flow (Skip validation)
                            raw_flow = self.get_flow_prediction(p_from, p_to)
                            if raw_flow is not None:
                                current_flow_predictions[f"{p_from}_{p_to}"] = raw_flow
                                found_count += 1
                                logging.info(f"✅ Recalculated flow (Skip validation): {p_from}->{p_to}")
                            else:
                                logging.error(f"❌ Failed to recalculate flow: {p_from}->{p_to}")
                        except Exception as e:
                            logging.error(f"❌ Error recalculating flow {p_from}->{p_to}: {e}")

                logging.info(f"Current window flow status: Found={found_count}, Missing={missing_count}, Available flows={len(current_flow_predictions)}")
                
                if len(current_flow_predictions) == 0:
                    logging.warning(f"⚠️ Window {start_idx}-{end_idx-1} has no available flow predictions, skipping optimization")
                    continue

                self.optimize_simple_flow(current_window_pairs, current_flow_predictions)

                if start_idx > 0 and window_size > 1: # Avoid freezing when there is only one window or single frame
                    self.freeze_frame(start_idx)
            
            self.unfreeze_frames()
            logging.info("Dense optical flow optimization completed")
            
            # Analyze depth changes after optimization
            if depth_before is not None and hasattr(self, 'reproj_depths_param'):
                depth_after = self.reproj_depths_param.detach().clone()
                self.analyze_depth_changes(depth_before, depth_after)
                
                # Calculate key statistics
                depth_diff = depth_after - depth_before
                mean_change = depth_diff.mean().item()
                std_change = depth_diff.std().item()
                max_change = depth_diff.max().item()
                min_change = depth_diff.min().item()
                
                logging.info(f"=== Optical Flow Optimization Depth Change Summary ===")
                logging.info(f"Mean depth change: {mean_change:.6f}")
                logging.info(f"Depth change std dev: {std_change:.6f}")
                logging.info(f"Max depth change: {max_change:.6f}")
                logging.info(f"Min depth change: {min_change:.6f}")
                
                # Issue warning if change is too large
                if abs(mean_change) > 0.5 or std_change > 2.0:
                    logging.warning("Depth change too large, may affect final result quality")
                    logging.warning("Suggest reducing flow weight or learning rate")

        except Exception as e:
            logging.error(f"Error in dense optical flow optimization: {e}")
            import traceback
            logging.error(traceback.format_exc())
            logging.warning("Skipping dense optical flow optimization")


    def constrain_depth_values(self):
        """
        Constrain depth values within a reasonable range to prevent excessive deviation from initial values
        """
        if not hasattr(self, 'reproj_depths_param'):
            return
            
        with torch.no_grad():
            # First check if parameters contain outliers
            if torch.isnan(self.reproj_depths_param).any():
                logging.error("Depth parameters contain NaN values, resetting")
                if hasattr(self, 'reproj_depths'):
                    self.reproj_depths_param.data.copy_(self.reproj_depths)
                return
                
            if torch.isinf(self.reproj_depths_param).any():
                logging.error("Depth parameters contain infinite values, resetting")
                if hasattr(self, 'reproj_depths'):
                    self.reproj_depths_param.data.copy_(self.reproj_depths)
                return
            
            # Get initial depth statistics (if not already present)
            if not hasattr(self, '_initial_depth_stats'):
                original_depth = self.reproj_depths.detach()
                mean_depth = original_depth.mean().item()
                std_depth = original_depth.std().item()
                
                # Use more conservative constraint range
                self._initial_depth_stats = {
                    'mean': mean_depth,
                    'std': std_depth,
                    'min': max(0.01, mean_depth - 1.5*std_depth),  # 1.5 sigma range, more conservative
                    'max': min(100.0, mean_depth + 1.5*std_depth)  # Limit maximum depth to 100
                }
                logging.info(f"Initial depth stats: Mean={mean_depth:.3f}, Std={std_depth:.3f}")
                logging.info(f"Depth constraint range: [{self._initial_depth_stats['min']:.3f}, {self._initial_depth_stats['max']:.3f}]")
            
            # Constrain depth values
            stats = self._initial_depth_stats
            old_min = self.reproj_depths_param.min().item()
            old_max = self.reproj_depths_param.max().item()
            
            self.reproj_depths_param.data.clamp_(stats['min'], stats['max'])
            
            new_min = self.reproj_depths_param.min().item()
            new_max = self.reproj_depths_param.max().item()
            
            # Log if values are constrained
            if old_min < stats['min'] or old_max > stats['max']:
                logging.debug(f"Depth values constrained: [{old_min:.3f}, {old_max:.3f}] -> [{new_min:.3f}, {new_max:.3f}]")


    def visualize_dyn_regions(self):
        """
        Visualize dynamic region optimization areas, showing only DeVA dynamic mask regions (label=1)
        without any involvement of EPI masks.
        """
        import cv2
        import os
        import numpy as np
        from tqdm import tqdm

        # Create save directory
        save_dir = os.path.join(self.opt.output_dir, "dyn_regions")
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Dynamic region visualizations will be saved to: {save_dir}")
        
        # Process each frame
        for f in tqdm(range(min(self.num_frames, len(self.dyn_masks_filters))), desc="Generating dynamic region visualization"):
            if hasattr(self, 'images') and self.images is not None:
                # Get current frame image
                orig_img = self.images[f].cpu().numpy()
                if orig_img.dtype != np.uint8:
                    orig_img = (orig_img * 255).astype(np.uint8)
                
                # Create image copy for visualization
                visualization = orig_img.copy()
                
                # Get dynamic region mask - ONLY DeVA dynamic areas (value=1)
                # Explicitly ignoring EPI dynamic areas (value=3)
                if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
                    # Use pure DeVA dynamic regions mask unaffected by EPI
                    dyn_mask = (self.dyn_masks_filters_deva[f] == 1)
                    
                    # Count pixel statistics
                    dyn_pixels = np.sum(dyn_mask)
                    total_pixels = dyn_mask.size
                    
                    logging.info(f"Frame {f} - DeVA dynamic region: {dyn_pixels} pixels ({dyn_pixels/total_pixels*100:.2f}%)")
                    
                    # Create color overlay for dynamic regions (red)
                    dyn_overlay = np.zeros_like(visualization)
                    dyn_overlay[dyn_mask] = [0, 0, 255]  # BGR format, red
                    
                    # Add text annotation
                    cv2.putText(dyn_overlay, f"DeVA dynamic regions only (label=1)", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Blend overlay with original image
                    alpha = 0.5
                    cv2.addWeighted(dyn_overlay, alpha, visualization, 1, 0, visualization)
                    
                    # Save visualization image
                    viz_path = os.path.join(save_dir, f"frame_{f:04d}_dyn_overlay.png")
                    cv2.imwrite(viz_path, visualization)
                    
                    # Save dynamic region mask image
                    dyn_mask_img = np.zeros((self.height, self.width), dtype=np.uint8)
                    dyn_mask_img[dyn_mask] = 255
                    mask_path = os.path.join(save_dir, f"frame_{f:04d}_dyn_mask.png")
                    cv2.imwrite(mask_path, dyn_mask_img)
        
        logging.info(f"Completed dynamic region mask visualization for all frames (DeVA dynamic only, label=1)")

    def check_epi_masks(self):
        """
        Diagnostic function to check and visualize EPI mask files.
        This helps debug issues with EPI mask loading.
        """
        import os
        import glob
        import imageio
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        # Determine EPI mask directory
        epi_mask_path = os.path.join(self.opt.BASE, "epi_mask")
        if not os.path.exists(epi_mask_path):
            logging.error(f"EPI mask directory not found: {epi_mask_path}")
            
            # Look for alternative directories that might contain masks
            base_dir = self.opt.BASE
            potential_dirs = glob.glob(os.path.join(base_dir, "*mask*")) + \
                            glob.glob(os.path.join(base_dir, "*epi*"))
            
            if potential_dirs:
                logging.info(f"Found potential alternative directories: {potential_dirs}")
            else:
                logging.info(f"No potential mask directories found in {base_dir}")
            return
        
        # List all files in the EPI mask directory
        mask_files = sorted(glob.glob(os.path.join(epi_mask_path, "*.png")))
        logging.info(f"Found {len(mask_files)} PNG files in {epi_mask_path}")
        
        if not mask_files:
            logging.error("No mask files found in the directory")
            return
        
        # Create output directory for diagnostics
        diag_dir = os.path.join(self.opt.output_dir, "epi_mask_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        
        # Process a sample of files (up to 5) to analyze
        sample_files = mask_files[:min(5, len(mask_files))]
        for idx, file_path in enumerate(sample_files):
            try:
                # Load the mask image
                mask_img = imageio.imread(file_path)
                file_name = os.path.basename(file_path)
                
                # Gather statistics
                shape = mask_img.shape
                dtype = mask_img.dtype
                min_val = np.min(mask_img)
                max_val = np.max(mask_img)
                unique_vals = np.unique(mask_img)
                n_unique = len(unique_vals)
                non_zero = np.sum(mask_img > 0)
                percent_non_zero = (non_zero / mask_img.size) * 100
                
                # Log information
                logging.info(f"Mask file: {file_name}")
                logging.info(f"  Shape: {shape}, Dtype: {dtype}")
                logging.info(f"  Value range: [{min_val}, {max_val}]")
                logging.info(f"  Unique values: {unique_vals[:10]}{'...' if n_unique > 10 else ''} ({n_unique} total)")
                logging.info(f"  Non-zero pixels: {non_zero} ({percent_non_zero:.2f}%)")
                
                # Generate visualizations
                plt.figure(figsize=(15, 10))
                
                # Original image visualization
                plt.subplot(2, 2, 1)
                if len(shape) == 3 and shape[2] == 3:  # Color image
                    plt.imshow(mask_img)
                else:  # Grayscale
                    plt.imshow(mask_img, cmap='gray')
                plt.title(f"Original Mask: {file_name}")
                plt.colorbar()
                
                # Binary thresholded versions
                thresholds = [0, 10, 50]
                for i, threshold in enumerate(thresholds):
                    binary = mask_img > threshold
                    plt.subplot(2, 2, i+2)
                    plt.imshow(binary, cmap='gray')
                    plt.title(f"Threshold > {threshold}: {np.sum(binary)} pixels ({np.sum(binary)/binary.size*100:.2f}%)")
                
                # Save diagnostic image
                plt.tight_layout()
                plt.savefig(os.path.join(diag_dir, f"diag_{idx+1}_{file_name}"))
                plt.close()
                
                # If original image has content, save brightened version
                if non_zero > 0 and max_val < 200:
                    # Try to enhance visibility of low-value masks
                    enhanced = mask_img.astype(np.float32)
                    if max_val > 0:
                        enhanced = enhanced * (255.0 / max_val)
                    enhanced = enhanced.astype(np.uint8)
                    cv2.imwrite(os.path.join(diag_dir, f"enhanced_{idx+1}_{file_name}"), enhanced)
                
            except Exception as e:
                logging.error(f"Error processing mask file {file_path}: {e}")
        
        logging.info(f"EPI mask diagnostics saved to {diag_dir}")
        
        # Also check expected file pattern matching
        frame_patterns = []
        for i in range(1, 6):  # Check first 5 frames
            pattern = f"frame_{i:04d}_dynamic_mask.png"
            full_path = os.path.join(epi_mask_path, pattern)
            exists = os.path.exists(full_path)
            frame_patterns.append((pattern, exists))
            logging.info(f"Expected mask file {pattern}: {'EXISTS' if exists else 'MISSING'}")
        
        return diag_dir

    def visualize_init_regions(self):
        """
        Visualize regions in initial optimization phase, using different colors for different region types
        """
        import cv2
        import os
        import numpy as np
        from tqdm import tqdm

        # Create save directory
        save_dir = os.path.join(self.opt.output_dir, "init_regions")
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving initialization optimization region visualization to: {save_dir}")
        
        # Check if masks are created
        if not hasattr(self, 'dyn_masks_filters') or self.dyn_masks_filters is None:
            logging.error("dyn_masks_filters not found, cannot visualize")
            return
        
        # Process each frame
        total_frames = min(self.num_frames, len(self.dyn_masks_filters))
        logging.info(f"Starting generating region visualization for {total_frames} frames")
        
        # Iterate through all frames
        for f in tqdm(range(total_frames), desc="Generating init region visualization"):
            if hasattr(self, 'images') and self.images is not None:
                # Get current frame image
                orig_img = self.images[f].cpu().numpy()
                if orig_img.dtype != np.uint8:
                    orig_img = (orig_img * 255).astype(np.uint8)
                
                # Get current frame mask
                mask = self.dyn_masks_filters[f]
                
                # Create color visualization image
                visualization = orig_img.copy()
                
                # Mark different regions with different colors
                # Use distinct colors to differentiate regions
                static_color = np.array([0, 255, 0])     # Green - Static region (0)
                deva_dyn_color = np.array([0, 0, 255])   # Red - DeVA dynamic region (1)
                border_color = np.array([128, 128, 128]) # Gray - Boundary/rejected region (2)
                epi_dyn_color = np.array([255, 0, 255])  # Purple - EPI dynamic region (3)
                
                # Count pixels for each region type
                static_mask = (mask == 0)
                deva_dyn_mask = (mask == 1)
                border_mask = (mask == 2)
                epi_dyn_mask = (mask == 3)
                
                static_pixels = np.sum(static_mask)
                deva_dyn_pixels = np.sum(deva_dyn_mask)
                border_pixels = np.sum(border_mask)
                epi_dyn_pixels = np.sum(epi_dyn_mask)
                total_pixels = mask.size
                
                logging.info(f"Frame {f} region stats: "
                            f"Static={static_pixels}({static_pixels/total_pixels*100:.2f}%), "
                            f"DeVA Dynamic={deva_dyn_pixels}({deva_dyn_pixels/total_pixels*100:.2f}%), "
                            f"Border={border_pixels}({border_pixels/total_pixels*100:.2f}%), "
                            f"EPI Dynamic={epi_dyn_pixels}({epi_dyn_pixels/total_pixels*100:.2f}%)")
                
                # Prepare color mask
                color_mask = np.zeros_like(orig_img)
                color_mask[static_mask] = static_color
                color_mask[deva_dyn_mask] = deva_dyn_color
                color_mask[border_mask] = border_color
                color_mask[epi_dyn_mask] = epi_dyn_color
                
                # Blend images
                alpha = 0.5
                overlay = cv2.addWeighted(orig_img, 1.0, color_mask, alpha, 0)
                
                # Add legend explanation
                legend_height = 30
                legend_img = np.ones((legend_height*4, overlay.shape[1], 3), dtype=np.uint8) * 255
                
                # Draw legend for each region
                legend_img[0:legend_height, :] = static_color
                cv2.putText(legend_img, "Static Region (Label 0)", (10, legend_height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                legend_img[legend_height:legend_height*2, :] = deva_dyn_color
                cv2.putText(legend_img, "DeVA Dynamic Region (Label 1)", (10, legend_height*2-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                legend_img[legend_height*2:legend_height*3, :] = border_color
                cv2.putText(legend_img, "Boundary/Rejected Region (Label 2)", (10, legend_height*3-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                legend_img[legend_height*3:legend_height*4, :] = epi_dyn_color
                cv2.putText(legend_img, "EPI Dynamic Region (Label 3)", (10, legend_height*4-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Merge legend with overlay image
                combined_img = np.vstack((overlay, legend_img))
                
                # Save visualization image
                viz_path = os.path.join(save_dir, f"frame_{f:04d}_regions_overlay.png")
                cv2.imwrite(viz_path, combined_img)
                
                # Optional: Save individual mask images for each region
                if f == 0:  # Only save individual masks for the first frame to save space
                    # Static region mask
                    static_img = np.zeros((self.height, self.width), dtype=np.uint8)
                    static_img[static_mask] = 255
                    cv2.imwrite(os.path.join(save_dir, f"frame_{f:04d}_static_mask.png"), static_img)
                    
                    # DeVA dynamic region mask
                    deva_dyn_img = np.zeros((self.height, self.width), dtype=np.uint8)
                    deva_dyn_img[deva_dyn_mask] = 255
                    cv2.imwrite(os.path.join(save_dir, f"frame_{f:04d}_deva_dyn_mask.png"), deva_dyn_img)
                    
                    # EPI dynamic region mask
                    epi_dyn_img = np.zeros((self.height, self.width), dtype=np.uint8)
                    epi_dyn_img[epi_dyn_mask] = 255
                    cv2.imwrite(os.path.join(save_dir, f"frame_{f:04d}_epi_dyn_mask.png"), epi_dyn_img)
                    
                    # Overlayed class mask (color coded)
                    color_coding = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    color_coding[static_mask] = static_color
                    color_coding[deva_dyn_mask] = deva_dyn_color
                    color_coding[border_mask] = border_color
                    color_coding[epi_dyn_mask] = epi_dyn_color
                    cv2.imwrite(os.path.join(save_dir, f"frame_{f:04d}_color_coded_mask.png"), color_coding)
        
        logging.info(f"Completed region visualization for all frames, using color code: Green=Static, Red=DeVA Dynamic, Gray=Border/Rejected, Purple=EPI Dynamic")

    def fuse_with_deva_masks(self, save_path, alpha=0.3):
        """
        Fuse depth point cloud while creating version with DeVA masks
        Save two sets of results: 1. Standard static + dynamic point cloud, 2. Point cloud with DeVA masks (each dynamic instance marked with a different color)
        
        Args:
            save_path: Save path
            alpha: Transparency of dynamic region, default 0.3
            
        Returns:
            tuple: (standard point cloud path, masked point cloud path)
        """
        import os
        import cv2
        import numpy as np

        # Create save directory
        mask_save_dir = os.path.join(os.path.dirname(save_path), "masks")
        os.makedirs(mask_save_dir, exist_ok=True)
        logging.info(f"Mask visualizations will be saved to: {mask_save_dir}")

        # Ensure coordinates are on the correct device
        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2), device=self.device)
        coords_2d_homo = torch.cat((coords, torch.ones((len(coords),1), device=self.device)), dim=1)  # N x 3
        
        # Create copy of original images
        images_original = self.images.clone().to(self.device)  # Ensure original images are on the correct device
        
        # Initialize point cloud storage arrays
        # 1. Standard point cloud
        static_points = np.zeros((0, 3))  # Static point cloud
        static_rgbs = np.zeros((0, 3), np.uint8)  # Static point cloud color
        dyn_points = []  # Dynamic point cloud list (by frame)
        dyn_rgbs = []   # Dynamic point cloud color list (by frame)
        
        # 2. Point cloud with masks
        masked_static_points = np.zeros((0, 3))  # Masked static point cloud
        masked_static_rgbs = np.zeros((0, 3), np.uint8)  # Masked static point cloud color
        masked_dyn_points = []  # Masked dynamic point cloud list (by frame)
        masked_dyn_rgbs = []   # Masked dynamic point cloud color list (by frame)
        
        # Get camera parameters
        Rs, ts = self.get_poses(torch.arange(self.num_frames, device=self.device))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames, device=self.device))
        
        # Check if dyn_masks_filters_deva exists
        if not hasattr(self, 'dyn_masks_filters_deva'):
            logging.warning("dyn_masks_filters_deva does not exist, will try to use dyn_masks_filters")
            if hasattr(self, 'dyn_masks_filters'):
                logging.info("Found dyn_masks_filters, using it as alternative")
                self.dyn_masks_filters_deva = self.dyn_masks_filters
            else:
                logging.warning("dyn_masks_filters also does not exist, using dyn_masks")
                # Create simple binary mask
                h, w = self.shape
                num_frames = self.dyn_masks.shape[0]
                self.dyn_masks_filters_deva = np.zeros((num_frames, h, w), dtype=np.uint8)
                for f in range(num_frames):
                    mask = self.dyn_masks[f].cpu().numpy()
                    self.dyn_masks_filters_deva[f] = (mask > 0).astype(np.uint8)
        
        # Find all instance IDs (assuming dyn_masks contains instance IDs)
        instance_ids = set()
        if hasattr(self, 'dyn_masks'):
            # Use original dyn_masks to get all instance IDs
            for f in range(self.num_frames):
                unique_ids = torch.unique(self.dyn_masks[f]).cpu().numpy()
                for id in unique_ids:
                    if id > 0:  # Exclude background (ID=0)
                        instance_ids.add(int(id))
        
        if not instance_ids:
            # If instance IDs cannot be found, use default value
            instance_ids = {1}  # Default only one instance (ID=1)
            logging.warning(f"No valid instance IDs found, using default ID: {instance_ids}")
        else:
            logging.info(f"Found the following instance IDs: {instance_ids}")
        
        # Assign a unique color to each instance ID
        instance_colors = {}
        
        # Use fixed color scheme (Use RGB format, matching expected final display color directly)
        fixed_colors = [
            [255, 0, 0],    # Instance 1: Red
            [0, 255, 0],    # Instance 2: Green
            [0, 0, 255]     # Instance 3: Blue
        ]
        
        # Assign colors to instances
        import random
        for instance_id in sorted(instance_ids):
            if instance_id <= 3:
                # First three instances use fixed colors
                color_idx = instance_id - 1
                instance_colors[instance_id] = np.array(fixed_colors[color_idx])
            else:
                # Randomly generate colors for other instances (RGB format)
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)
                instance_colors[instance_id] = np.array([r, g, b])
        
        logging.info(f"Assigned colors to {len(instance_colors)} instances")
        for instance_id, color in instance_colors.items():
            color_name = ""
            if np.array_equal(color, [255, 0, 0]):
                color_name = "Red"
            elif np.array_equal(color, [0, 255, 0]):
                color_name = "Green"
            elif np.array_equal(color, [0, 0, 255]):
                color_name = "Blue"
            else:
                color_name = f"RGB({color[0]},{color[1]},{color[2]})"
            logging.info(f"  Instance {instance_id}: {color_name} {color}")
        
        logging.info(f"Starting Processing point cloud and DeVA masks...")
        for f in range(self.num_frames):
            # Ensure all tensors are on the same device
            R = Rs[f].to(self.device)
            t = ts[f].to(self.device)
            K = Ks[f].to(self.device)

            # Use torch.matmul instead of @ operator to explicitly represent matrix multiplication
            src_ray = torch.matmul(torch.linalg.inv(K), coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            # Get depth map
            depth = self.reproj_depths[f].to(self.device)  # Ensure depth on correct device
            # Apply edge filtering
            edge_mask = self.filter_depth_edges(depth)
            edge_mask = edge_mask.flatten()  # NumPy array
            
            # --- Get DeVA mask ---
            # Use dyn_masks_filters_deva to get DeVA mask
            # Label 0: Static region
            # Label 1: DeVA dynamic region
            current_deva_mask = self.dyn_masks_filters_deva[f].flatten()
            static_mask = (current_deva_mask == 0)  # Static region
            dyn_mask = (current_deva_mask == 1)     # DeVA dynamic region
            
            # Apply edge filtering
            static_mask = np.logical_and(static_mask, edge_mask)
            dyn_mask = np.logical_and(dyn_mask, edge_mask)
            
            # If processing multiple instances is needed, use dyn_masks to get instance IDs
            instance_masks = {}
            if len(instance_ids) > 1 and hasattr(self, 'dyn_masks'):
                current_dyn_mask_f_np = self.dyn_masks[f].cpu().numpy().flatten()
                for instance_id in instance_ids:
                    # Create mask for each instance ID
                    instance_mask = (current_dyn_mask_f_np == instance_id)
                    # Ensure the instance is within DeVA dynamic region and passes edge filtering
                    instance_mask = np.logical_and(instance_mask, dyn_mask)
                    if np.any(instance_mask):
                        instance_masks[instance_id] = instance_mask
            else:
                # If there is only one instance or no instance info, treat all dynamic regions as one instance
                if 1 in instance_ids:
                    instance_masks[1] = dyn_mask
                else:
                    instance_id = next(iter(instance_ids))
                    instance_masks[instance_id] = dyn_mask
            
            # Save visualization masks (only for representative frames)
            if f % 10 == 0:  # Save mask visualization every 10 frames
                # Create mask image
                mask_img = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
                # Static region - Green
                mask_img[static_mask.reshape(self.shape)] = [0, 255, 0]
                # DeVA dynamic region - Red
                mask_img[dyn_mask.reshape(self.shape)] = [0, 0, 255]
                
                # Save mask image
                cv2.imwrite(os.path.join(mask_save_dir, f"frame_{f:04d}_deva_mask.png"), mask_img)
            
            # Calculate 3D point cloud coordinates - ensure all calculations are on the same device
            points_3d_src = src_ray_homo * depth.flatten().unsqueeze(-1)
            points_3d_world = torch.matmul(R, points_3d_src.T).T + t.squeeze(-1)
            
            # Get RGB colors - ensure on the same device
            rgb = images_original[f].reshape((-1, 3))
            
            # Convert mask to pytorch tensor and place on correct device
            static_mask_tensor = torch.from_numpy(static_mask).to(self.device)
            dyn_mask_tensor = torch.from_numpy(dyn_mask).to(self.device)
            
            # === Processing Static Regions ===
            if static_mask_tensor.sum() > 0:
                points_3d_world_static = points_3d_world[static_mask_tensor]
                rgb_static = rgb[static_mask_tensor]
                
                # Add to standard static point cloud
                static_points = np.vstack([static_points, points_3d_world_static.detach().cpu().numpy()])
                static_rgbs = np.vstack([static_rgbs, rgb_static.detach().cpu().numpy()])
                
                # Add to masked static point cloud
                masked_static_points = np.vstack([masked_static_points, points_3d_world_static.detach().cpu().numpy()])
                masked_static_rgbs = np.vstack([masked_static_rgbs, rgb_static.detach().cpu().numpy()])
            
            # === Processing Dynamic Regions ===
            # Initialize dynamic point cloud for current frame
            frame_dyn_points = []
            frame_dyn_rgbs = []
            frame_masked_dyn_points = []
            frame_masked_dyn_rgbs = []
            
            # If processing multiple instances is needed, use dyn_masks to get instance IDs
            if len(instance_ids) > 1 and hasattr(self, 'dyn_masks'):
                current_dyn_mask_f_np = self.dyn_masks[f].cpu().numpy().flatten()
                for instance_id in instance_ids:
                    # Create mask for each instance ID
                    instance_mask = (current_dyn_mask_f_np == instance_id)
                    # Ensure the instance is within DeVA dynamic region and passes edge filtering
                    instance_mask = np.logical_and(instance_mask, dyn_mask)
                    
                    if np.any(instance_mask):
                        # Convert instance mask to tensor and put on correct device
                        instance_mask_tensor = torch.from_numpy(instance_mask).to(self.device)
                        
                        # Get point cloud for current instance
                        instance_points = points_3d_world[instance_mask_tensor]
                        instance_rgbs = rgb[instance_mask_tensor]
                        
                        # Standard dynamic point cloud (keep original color)
                        frame_dyn_points.extend(instance_points.detach().cpu().numpy())
                        frame_dyn_rgbs.extend(instance_rgbs.detach().cpu().numpy())
                        
                        # Masked dynamic point cloud (apply color blending)
                        instance_color = instance_colors.get(instance_id, np.array([0, 0, 255]))  # Default Blue
                        instance_rgbs_np = instance_rgbs.detach().cpu().numpy()
                        blended_colors = instance_rgbs_np * (1-alpha) + instance_color * alpha
                        
                        frame_masked_dyn_points.extend(instance_points.detach().cpu().numpy())
                        frame_masked_dyn_rgbs.extend(blended_colors.astype(np.uint8))
            else:
                # If there is only one instance or no instance info, treat all dynamic regions as one instance
                if dyn_mask_tensor.sum() > 0:
                    points_3d_world_dyn = points_3d_world[dyn_mask_tensor]
                    rgb_dyn = rgb[dyn_mask_tensor]
                    
                    # Standard dynamic point cloud (keep original color)
                    frame_dyn_points.extend(points_3d_world_dyn.detach().cpu().numpy())
                    frame_dyn_rgbs.extend(rgb_dyn.detach().cpu().numpy())
                    
                    # Masked dynamic point cloud (apply color blending)
                    instance_id = 1 if 1 in instance_ids else next(iter(instance_ids))
                    instance_color = instance_colors.get(instance_id, np.array([0, 0, 255]))  # Default Blue
                    rgb_dyn_np = rgb_dyn.detach().cpu().numpy()
                    blended_colors = rgb_dyn_np * (1-alpha) + instance_color * alpha
                    
                    frame_masked_dyn_points.extend(points_3d_world_dyn.detach().cpu().numpy())
                    frame_masked_dyn_rgbs.extend(blended_colors.astype(np.uint8))
            
            # Add current frame's dynamic point cloud to list
            if frame_dyn_points:
                dyn_points.append(np.array(frame_dyn_points))
                dyn_rgbs.append(np.array(frame_dyn_rgbs))
            else:
                dyn_points.append(np.zeros((0, 3)))
                dyn_rgbs.append(np.zeros((0, 3), dtype=np.uint8))
                
            if frame_masked_dyn_points:
                masked_dyn_points.append(np.array(frame_masked_dyn_points))
                masked_dyn_rgbs.append(np.array(frame_masked_dyn_rgbs))
            else:
                masked_dyn_points.append(np.zeros((0, 3)))
                masked_dyn_rgbs.append(np.zeros((0, 3), dtype=np.uint8))
            
            if f % 10 == 0:
                logging.info(f"Processed {f}/{self.num_frames} frames")
                logging.info(f"  Static point count: {static_mask_tensor.sum().item()}")
                logging.info(f"  Dynamic point count: {len(frame_dyn_points)}")
        
        # Prepare camera poses
        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()
        
        logging.info(f"\nPoint Cloud Processing Completed:")
        logging.info(f"Total standard static points: {len(static_points)}")
        logging.info(f"Standard dynamic point cloud frames: {len(dyn_points)}")
        logging.info(f"Total masked static points: {len(masked_static_points)}")
        logging.info(f"Masked dynamic point cloud frames: {len(masked_dyn_points)}")
        
        # Save standard point cloud data
        standard_output = {
            "static_points": static_points,
            "static_rgbs": static_rgbs,
            "dyn_points": np.array(dyn_points, dtype=object),
            "dyn_rgbs": np.array(dyn_rgbs, dtype=object),
            "c2w": c2w,
            "rgbs": images_original.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu()
        }
        
        # Save masked point cloud data
        masked_output = {
            "static_points": masked_static_points,  # Background static points
            "static_rgbs": masked_static_rgbs,      # Background static point colors
            "dyn_points": np.array(masked_dyn_points, dtype=object),  # Dynamic instance points (by frame)
            "dyn_rgbs": np.array(masked_dyn_rgbs, dtype=object),      # Dynamic instance point colors (by frame)
            "c2w": c2w,
            "rgbs": images_original.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu(),
            "instance_colors": instance_colors  # Save instance color map for future use
        }
        
        # Save both versions of point clouds
        standard_path = save_path
        masked_path = os.path.join(os.path.dirname(save_path), "deva_masked_" + os.path.basename(save_path))
        
        logging.info(f"Saving standard point cloud data to: {standard_path}")
        np.savez_compressed(standard_path, **standard_output)
        
        logging.info(f"Saving point cloud data with DeVA masks to: {masked_path}")
        np.savez_compressed(masked_path, **masked_output)
        
        return standard_path, masked_path
    
    def validate_flow_depth_consistency(self, flow_pred, from_idx, to_idx):
        """
        Validate the consistency of flow map and depth map dimensions to ensure correct pixel coordinate correspondence.
        
        Args:
            flow_pred: Flow prediction result (H, W, 2)
            from_idx: Source frame index
            to_idx: Target frame index
            
        Returns:
            validated_flow: Validated and corrected flow (H, W, 2)
        """
        if flow_pred is None:
            logging.error(f"Flow validation: Frame {from_idx}->{to_idx} flow prediction is None")
            return None
            
        # Get depth map dimensions
        depth_h, depth_w = self.shape
        flow_h, flow_w = flow_pred.shape[:2]
        
        logging.info(f"Flow validation: Frame {from_idx}->{to_idx}, Depth map size: {depth_h}x{depth_w}, Flow size: {flow_h}x{flow_w}")
        
        if flow_h != depth_h or flow_w != depth_w:
            logging.warning(f"Flow and depth map dimensions do not match! Correcting...")
            
            # Interpolate flow to the correct size
            flow_tensor = flow_pred.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
            flow_tensor = F.interpolate(flow_tensor, size=(depth_h, depth_w), mode='bilinear', align_corners=True)
            
            # Scale flow values proportionally
            scale_h = depth_h / flow_h
            scale_w = depth_w / flow_w
            flow_tensor[0, 0] *= scale_w  # x-direction flow
            flow_tensor[0, 1] *= scale_h  # y-direction flow
            
            validated_flow = flow_tensor.squeeze(0).permute(1, 2, 0)  # Convert back to (H, W, 2)
            
            logging.info(f"Flow corrected to depth map size {depth_h}x{depth_w}, scale factors: x={scale_w:.3f}, y={scale_h:.3f}")
            
            # Verify corrected dimensions
            final_h, final_w = validated_flow.shape[:2]
            if final_h != depth_h or final_w != depth_w:
                logging.error(f"Flow correction failed! Corrected size: {final_h}x{final_w}, Expected: {depth_h}x{depth_w}")
                return None
                
        else:
            validated_flow = flow_pred
            logging.info(f"Flow dimension validation passed: {flow_h}x{flow_w}")
        
        # Additional validation: Check rationality of flow values
        flow_magnitude = torch.norm(validated_flow, dim=2)
        max_flow = flow_magnitude.max().item()
        mean_flow = flow_magnitude.mean().item()
        
        # Flow values should not be excessively large (generally not exceeding half the image size)
        reasonable_max = max(depth_h, depth_w) * 0.5
        if max_flow > reasonable_max:
            logging.warning(f"Flow values might be too large: max={max_flow:.2f}, image size: {depth_h}x{depth_w}")
            
        logging.info(f"Flow statistics: max={max_flow:.2f}, mean={mean_flow:.2f}")
        
        return validated_flow
    
    def check_flow_optimization_inputs(self):
        """
        Check if all input data required for flow optimization is valid.
        Returns whether it is safe to proceed with flow optimization.
        """
        logging.info("Checking flow optimization input data...")
        
        # Check depth parameters
        if not hasattr(self, 'reproj_depths_param') or self.reproj_depths_param is None:
            logging.error("reproj_depths_param does not exist")
            return False
        
        # Check if depth parameters contain NaN or infinite values
        if torch.isnan(self.reproj_depths_param).any():
            logging.error("reproj_depths_param contains NaN values")
            return False
            
        if torch.isinf(self.reproj_depths_param).any():
            logging.error("reproj_depths_param contains infinite values")
            return False
        
        # Check if depth values are reasonable
        min_depth = self.reproj_depths_param.min().item()
        max_depth = self.reproj_depths_param.max().item()
        mean_depth = self.reproj_depths_param.mean().item()
        
        if min_depth <= 0:
            logging.error(f"Depth values contain non-positive values: min={min_depth}")
            return False
            
        if max_depth > 1000:
            logging.warning(f"Depth values are unusually large: max={max_depth}")
            
        logging.info(f"Depth statistics: min={min_depth:.3f}, max={max_depth:.3f}, mean={mean_depth:.3f}")
        
        # Check camera poses
        try:
            Rs, ts = self.get_poses([0, 1])
            if torch.isnan(Rs).any() or torch.isinf(Rs).any():
                logging.error("Camera rotation matrices contain NaN or infinite values")
                return False
                
            if torch.isnan(ts).any() or torch.isinf(ts).any():
                logging.error("Camera translation vectors contain NaN or infinite values")
                return False
                
            # Check if rotation matrices are valid
            for i, R in enumerate(Rs):
                det_R = torch.linalg.det(R)
                if torch.abs(det_R - 1.0) > 0.1:
                    logging.warning(f"Determinant of rotation matrix {i} is abnormal: {det_R}")
                    
        except Exception as e:
            logging.error(f"Error getting camera poses: {e}")
            return False
        
        # Check camera intrinsics
        try:
            Ks = self.get_intrinsics_K([0])
            if torch.isnan(Ks).any() or torch.isinf(Ks).any():
                logging.error("Camera intrinsic matrices contain NaN or infinite values")
                return False
                
            # Check if intrinsic matrices are invertible
            det_K = torch.linalg.det(Ks[0])
            if torch.abs(det_K) < 1e-6:
                logging.error(f"Camera intrinsic matrix is nearly singular: det(K)={det_K}")
                return False
                
        except Exception as e:
            logging.error(f"Error getting camera intrinsics: {e}")
            return False
        
        # Check optimization mask
        try:
            interpolated_mask, preservation_mask = self.create_interpolated_static_optimization_mask()
            if interpolated_mask.sum() == 0:
                logging.warning("No interpolated static points requiring optimization")
                
            total_pixels = interpolated_mask.numel()
            opt_pixels = interpolated_mask.sum().item()
            pres_pixels = preservation_mask.sum().item()
            
            logging.info(f"Optimization region stats: optimization pixels={opt_pixels}/{total_pixels} ({opt_pixels/total_pixels*100:.2f}%)")
            logging.info(f"Preservation region stats: preservation pixels={pres_pixels}/{total_pixels} ({pres_pixels/total_pixels*100:.2f}%)")
            
        except Exception as e:
            logging.error(f"Error creating optimization mask: {e}")
            return False
        
        logging.info("✅ All input data checks passed, flow optimization can proceed")
        return True

    def BA_dense(self, from_idx=None, pixel_coords=None, static_mask=None, depth_values=None):
        """
        Perform BA optimization using dense interpolated static pixel points.
        
        Args:
            from_idx: Frame index
            pixel_coords: Pixel coordinates (N, 2)
            static_mask: Interpolated static mask (N,) - Co-tracker static points excluded
            depth_values: Depth values (N,)
            
        Returns:
            Optimization error
        """
        device = self.device
        
        if from_idx is None or pixel_coords is None or static_mask is None or depth_values is None:
            # If pixel data is not provided, return regular BA result
            logging.warning("BA_dense: Pixel data not provided, using sparse BA_fast instead")
            return self.BA_fast(is_dyn=False)
            
        # Ensure all tensors are on the same device
        pixel_coords = pixel_coords.to(device)
        static_mask = static_mask.to(device)
        depth_values = depth_values.to(device)
        
        # Only use pixels in static regions (Co-tracker static points excluded, only interpolated static points retained)
        valid_mask = static_mask
        if valid_mask.sum() == 0:
            logging.warning(f"BA_dense: Frame {from_idx} has no valid interpolated static pixels")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        valid_pixels = pixel_coords[valid_mask]
        valid_depths = depth_values[valid_mask]
        
        # Output statistics
        total_pixels = len(pixel_coords)
        valid_pixels_count = valid_mask.sum().item()
        logging.info(f"BA_dense: Frame {from_idx} using {valid_pixels_count}/{total_pixels} interpolated static pixels for dense BA optimization")
        
        # Get camera parameters
        Rs, ts = self.get_poses(torch.arange(self.num_frames).to(device))
        K = self.get_intrinsics_K(torch.arange(self.num_frames).to(device))
        
        # Ensure all tensors are on the same device
        Rs = Rs.to(device)
        ts = ts.to(device)
        K = K.to(device)
        
        # Construct pixel homogeneous coordinates
        valid_pixels_homo = torch.cat([valid_pixels, torch.ones((valid_pixels.shape[0], 1), device=device)], dim=1)
        
        # Reconstruct 3D points from pixel coordinates and depth values
        inv_K = torch.linalg.inv(K[from_idx])
        cam_rays = torch.matmul(inv_K, valid_pixels_homo.t()).t()
        cam_rays_norm = cam_rays / (cam_rays[:, 2].unsqueeze(-1) + 1e-16)
        
        # Reconstruct 3D points in world coordinate system
        points_3d = cam_rays_norm * valid_depths.unsqueeze(-1)
        R_from = Rs[from_idx]
        t_from = ts[from_idx]
        points_world = torch.matmul(R_from, points_3d.t()).t() + t_from.squeeze(-1)
        
        # Initialize Error
        total_error = torch.tensor(0.0, device=device, requires_grad=True)
        valid_frames = 0
        
        # Project these points to other frames and compute reprojection error
        for frame_idx in range(self.num_frames):
            if frame_idx == from_idx:
                continue  # Skipping source frame
                
            # Get camera parameters for current frame
            R_to = Rs[frame_idx]
            t_to = ts[frame_idx]
            K_to = K[frame_idx]
            
            # Transform points from world coordinate system to camera coordinate system
            points_cam = torch.matmul(torch.linalg.inv(R_to), (points_world - t_to.squeeze(-1)).t()).t()
            
            # Filter out points behind the camera
            cam_front_mask = points_cam[:, 2] > 0
            if cam_front_mask.sum() == 0:
                continue
                
            # Compute pixel coordinates
            points_cam_valid = points_cam[cam_front_mask]
            points_img = torch.matmul(K_to, points_cam_valid.t()).t()
            points_img = points_img / (points_img[:, 2].unsqueeze(-1) + 1e-16)
            projected_coords = points_img[:, :2]
            
            # Record original point indices for subsequent lookup of corresponding original pixel coordinates
            valid_indices = torch.where(valid_mask)[0][cam_front_mask]
            
            # Check if projected points are within image boundaries
            H, W = self.shape
            in_frame_mask = (
                (projected_coords[:, 0] >= 0) & 
                (projected_coords[:, 0] < W) &
                (projected_coords[:, 1] >= 0) & 
                (projected_coords[:, 1] < H)
            )
            
            if in_frame_mask.sum() == 0:
                continue
                
            # Filter out valid projected points and corresponding original indices
            valid_proj_coords = projected_coords[in_frame_mask]
            valid_indices = valid_indices[in_frame_mask]
            
            # Get static mask for that frame and exclude Co-tracker static points
            if hasattr(self, 'is_static_strict_tensor') and self.is_static_strict_tensor is not None:
                frame_static_mask = self.is_static_strict_tensor[frame_idx]
                
                # Exclude Co-tracker static points, keep only interpolated static points
                try:
                    cotracker_static_mask = self.get_cotracker_static_mask()
                    cotracker_mask_frame = cotracker_static_mask[frame_idx]
                    
                    # Interpolated static region = Strict static region - Co-tracker static points
                    interpolated_static_mask = frame_static_mask & (~cotracker_mask_frame.to(self.device))
                    frame_static_mask = interpolated_static_mask
                    
                except Exception as e:
                    logging.warning(f"BA_dense: Failed to get Co-tracker mask for frame {frame_idx}: {e}, using original static mask")
                
                # Check if projected points fall into interpolated static region
                proj_y = valid_proj_coords[:, 1].long().clamp(0, H-1)
                proj_x = valid_proj_coords[:, 0].long().clamp(0, W-1)
                
                static_proj_mask = frame_static_mask[proj_y, proj_x]
                
                if static_proj_mask.sum() == 0:
                    continue
                    
                # Only keep points projected to interpolated static regions
                valid_proj_coords = valid_proj_coords[static_proj_mask]
                valid_indices = valid_indices[static_proj_mask]
            
            # Calculate correspondences between two frames similar to flow model prediction
            if hasattr(self, 'flow_processor') and self.flow_processor is not None:
                # Try to get pre-calculated flow prediction
                flow_pred = None
                try:
                    # Get raw flow prediction
                    raw_flow_pred = self.get_flow_prediction(from_idx, frame_idx)
                    # Validate consistency between flow and depth map
                    flow_pred = self.validate_flow_depth_consistency(raw_flow_pred, from_idx, frame_idx)
                    if flow_pred is None:
                        logging.warning(f"BA_dense: Flow validation failed, frame {from_idx}->{frame_idx}")
                except Exception as e:
                    logging.warning(f"BA_dense: Failed to get flow prediction, frame {from_idx}->{frame_idx}: {e}")
                    flow_pred = None
                    
                if flow_pred is not None:
                    # If valid flow exists, use it
                    flow_pred = flow_pred.to(device)
                    
                    # Extract original coordinates of these points from source frame
                    orig_coords = pixel_coords[valid_indices]
                    
                    # Use flow prediction to get corresponding points in target frame
                    orig_y = orig_coords[:, 1].long().clamp(0, H-1)
                    orig_x = orig_coords[:, 0].long().clamp(0, W-1)
                    
                    # Get flow vectors from source frame to target frame
                    flow_vectors = flow_pred[orig_y, orig_x]
                    
                    # Calculate corresponding point coordinates in target frame
                    target_coords = orig_coords + flow_vectors
                    
                    # Ensure target coordinates are within image range
                    valid_target_mask = (
                        (target_coords[:, 0] >= 0) & 
                        (target_coords[:, 0] < W) & 
                        (target_coords[:, 1] >= 0) & 
                        (target_coords[:, 1] < H)
                    )
                    
                    if valid_target_mask.sum() > 0:
                        # Only keep points with valid target coordinates
                        valid_proj_coords = valid_proj_coords[valid_target_mask]
                        target_coords = target_coords[valid_target_mask]
                        
                        # Calculate difference between projected coordinates and target coordinates (similar to flow error)
                        reproj_error = flow_norm(valid_proj_coords - target_coords, self.flow_norm_l1)
                        
                        if torch.isnan(reproj_error).any():
                            logging.warning(f"BA_dense: Frame {frame_idx} contains NaN error")
                            continue
                            
                        # Average the errors of all points
                        total_error = total_error + reproj_error.mean()
                        valid_frames += 1
                        continue
            
            # If no flow model or prediction, compute error against projected point itself
            # In this case, we assume static points should have similar projected positions in different frames
            # Calculate reprojection error (3D points in static region should be consistent across frames)
            # Since there is no reference point, use an approximation: deviation from mean projection position
            mean_pos = valid_proj_coords.mean(dim=0, keepdim=True)
            error = flow_norm(valid_proj_coords - mean_pos, self.flow_norm_l1)
            
            if torch.isnan(error).any():
                logging.warning(f"BA_dense: Frame {frame_idx} contains NaN error")
                continue
                
            total_error = total_error + error.mean()
            valid_frames += 1
        
        # Calculate average error
        if valid_frames > 0:
            avg_error = total_error / valid_frames
        else:
            logging.warning("BA_dense: No valid frames for calculation")
            avg_error = torch.tensor(0.0, device=device, requires_grad=True)
            # Add a tiny gradient to ensure optimizer works
            if hasattr(self, 'active_optimizers'):
                for opt_name in self.active_optimizers:
                    if opt_name in self.optimizers:
                        for param_group in self.optimizers[opt_name].param_groups:
                            for param in param_group['params']:
                                if param.requires_grad:
                                    avg_error = avg_error + 0.0 * param.sum()
        
        return avg_error

    def create_interpolated_static_optimization_mask(self):
        """
        Create a precise mask only for optimizing interpolated static points.
        This mask identifies which pixels should undergo depth optimization and which should remain unchanged.
        
        Returns:
            interpolated_mask: [F, H, W] Boolean tensor, True indicates interpolated static points to be optimized
            preservation_mask: [F, H, W] Boolean tensor, True indicates cotracker points to be preserved
        """
        h, w = self.shape
        num_frames = self.num_frames
        
        # Initialize masks
        interpolated_mask = torch.zeros((num_frames, h, w), dtype=torch.bool, device=self.device)
        preservation_mask = torch.zeros((num_frames, h, w), dtype=torch.bool, device=self.device)
        
        # Get strict static region mask
        if hasattr(self, 'is_static_strict_tensor') and self.is_static_strict_tensor is not None:
            strict_static_mask = self.is_static_strict_tensor
        else:
            logging.warning("is_static_strict_tensor does not exist, using dyn_masks_filters")
            strict_static_mask = torch.zeros((num_frames, h, w), dtype=torch.bool, device=self.device)
            for f in range(num_frames):
                if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
                    static_mask_f = (self.dyn_masks_filters_deva[f] == 0)
                else:
                    static_mask_f = (self.dyn_masks_filters[f] == 0)
                strict_static_mask[f] = torch.from_numpy(static_mask_f).to(self.device)
        
        # Get cotracker static point mask
        try:
            cotracker_static_mask = self.get_cotracker_static_mask()
            if isinstance(cotracker_static_mask, torch.Tensor):
                cotracker_mask_tensor = cotracker_static_mask.to(self.device)
            else:
                cotracker_mask_tensor = torch.from_numpy(cotracker_static_mask).to(self.device)
            
            # Calculate interpolated static point mask: strict static region - cotracker static points
            interpolated_mask = strict_static_mask & (~cotracker_mask_tensor)
            
            # Preservation mask for cotracker points: strict static region ∩ cotracker static points
            preservation_mask = strict_static_mask & cotracker_mask_tensor
            
            # Statistics
            total_pixels = num_frames * h * w
            interpolated_pixels = interpolated_mask.sum().item()
            preservation_pixels = preservation_mask.sum().item()
            
            logging.info(f"Optimization mask creation completed:")
            logging.info(f"  Interpolated static points (to optimize): {interpolated_pixels} ({interpolated_pixels/total_pixels*100:.2f}%)")
            logging.info(f"  Cotracker static points (to preserve): {preservation_pixels} ({preservation_pixels/total_pixels*100:.2f}%)")
            
        except Exception as e:
            logging.warning(f"Error creating optimization mask: {e}")
            logging.warning("Will use entire strict static region as optimization target")
            interpolated_mask = strict_static_mask.clone()
            preservation_mask = torch.zeros_like(strict_static_mask)
        
        return interpolated_mask, preservation_mask

    def compute_depth_preservation_loss(self, optimization_mask, preservation_mask):
        """
        Compute depth preservation loss to ensure depth of cotracker points remains unchanged.
        
        Args:
            optimization_mask: Mask of regions to optimize
            preservation_mask: Mask of regions to preserve
            
        Returns:
            preservation_loss: Preservation loss
        """
        if not hasattr(self, 'reproj_depths_param') or not hasattr(self, 'reproj_depths'):
            return torch.tensor(0.0, device=self.device)
        
        # Get current depth and original depth
        current_depth = self.reproj_depths_param
        original_depth = self.reproj_depths.detach()
        
        # === Added: Numerical stability check ===
        if torch.isnan(current_depth).any() or torch.isinf(current_depth).any():
            logging.error("Current depth parameter contains NaN or infinite values")
            return torch.tensor(0.0, device=self.device)
            
        if torch.isnan(original_depth).any() or torch.isinf(original_depth).any():
            logging.error("Original depth parameter contains NaN or infinite values")
            return torch.tensor(0.0, device=self.device)
        
        # Ensure consistent dimensions
        if len(current_depth.shape) == 4 and current_depth.shape[1] == 1:
            current_depth = current_depth.squeeze(1)  # [F, H, W]
        if len(original_depth.shape) == 4 and original_depth.shape[1] == 1:
            original_depth = original_depth.squeeze(1)  # [F, H, W]
        
        # Check if preservation mask is valid
        if preservation_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate difference in preservation region
        depth_diff = torch.abs(current_depth - original_depth)
        
        # Check if difference is valid
        if torch.isnan(depth_diff).any() or torch.isinf(depth_diff).any():
            logging.error("Depth difference contains NaN or infinite values")
            return torch.tensor(0.0, device=self.device)
        
        preservation_loss = torch.mean(depth_diff[preservation_mask])
        
        # Check final loss validity
        if torch.isnan(preservation_loss) or torch.isinf(preservation_loss):
            logging.error("Depth preservation loss is NaN or infinite")
            return torch.tensor(0.0, device=self.device)
        
        return preservation_loss

    def compute_depth_smoothness_loss(self, optimization_mask):
        """
        Compute depth map smoothness loss, applied only in optimization regions.
        
        Args:
            optimization_mask: Optimization region mask
            
        Returns:
            smoothness_loss: Smoothness loss
        """
        if not hasattr(self, 'reproj_depths_param'):
            return torch.tensor(0.0, device=self.device)
        
        depth = self.reproj_depths_param
        
        # === Added: Numerical stability check ===
        if torch.isnan(depth).any() or torch.isinf(depth).any():
            logging.error("Depth parameter contains NaN or infinite values")
            return torch.tensor(0.0, device=self.device)
        
        if len(depth.shape) == 4:
            depth = depth.squeeze(1)  # [F, H, W]
        
        # Compute gradients - only within optimization regions
        smoothness_loss = torch.tensor(0.0, device=self.device)
        valid_frames = 0
        
        for f in range(depth.shape[0]):
            frame_mask = optimization_mask[f]
            if frame_mask.sum() == 0:
                continue
                
            frame_depth = depth[f]
            
            # Check if current frame depth is valid
            if torch.isnan(frame_depth).any() or torch.isinf(frame_depth).any():
                logging.warning(f"Frame {f} depth contains NaN or infinite values, skipping smoothness loss calculation")
                continue
            
            # Compute x-direction gradient (only on valid pixels)
            grad_x = torch.abs(frame_depth[:, :-1] - frame_depth[:, 1:])
            mask_x = frame_mask[:, :-1] & frame_mask[:, 1:]  # Ensure both adjacent pixels are in optimization region
            
            # Check x-direction gradient validity
            if torch.isnan(grad_x).any() or torch.isinf(grad_x).any():
                logging.warning(f"Frame {f} x-direction gradient contains NaN or infinite values")
            elif mask_x.sum() > 0:
                x_loss = torch.mean(grad_x[mask_x])
                if not torch.isnan(x_loss) and not torch.isinf(x_loss):
                    smoothness_loss += x_loss
            
            # Compute y-direction gradient (only on valid pixels)
            grad_y = torch.abs(frame_depth[:-1, :] - frame_depth[1:, :])
            mask_y = frame_mask[:-1, :] & frame_mask[1:, :]  # Ensure both adjacent pixels are in optimization region
            
            # Check y-direction gradient validity
            if torch.isnan(grad_y).any() or torch.isinf(grad_y).any():
                logging.warning(f"Frame {f} y-direction gradient contains NaN or infinite values")
            elif mask_y.sum() > 0:
                y_loss = torch.mean(grad_y[mask_y])
                if not torch.isnan(y_loss) and not torch.isinf(y_loss):
                    smoothness_loss += y_loss
                
            valid_frames += 1
        
        if valid_frames > 0:
            smoothness_loss = smoothness_loss / valid_frames
            
        # Final check if smoothness loss is valid
        if torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss):
            logging.error("Smoothness loss is NaN or infinite")
            return torch.tensor(0.0, device=self.device)
            
        return smoothness_loss

    def compute_depth_stability_loss(self, optimization_mask, initial_depth):
        """
        Compute depth stability loss to prevent excessive depth changes leading to d1 accuracy drop.
        Maintain stability by limiting the magnitude of change relative to initial depth.
        
        Args:
            optimization_mask: Optimization region mask
            initial_depth: Initial depth map
            
        Returns:
            stability_loss: Depth stability loss
        """
        if not hasattr(self, 'reproj_depths_param'):
            return torch.tensor(0.0, device=self.device)
        
        current_depth = self.reproj_depths_param
        
        # Numerical stability check
        if torch.isnan(current_depth).any() or torch.isinf(current_depth).any():
            logging.error("Current depth parameter contains NaN or infinite values")
            return torch.tensor(0.0, device=self.device)
        
        # Ensure consistent dimensions
        if len(current_depth.shape) == 4 and current_depth.shape[1] == 1:
            current_depth = current_depth.squeeze(1)  # [F, H, W]
        if len(initial_depth.shape) == 4 and initial_depth.shape[1] == 1:
            initial_depth = initial_depth.squeeze(1)  # [F, H, W]
        
        # Compute relative change - using relative error instead of absolute error
        relative_change = torch.abs(current_depth - initial_depth) / (initial_depth + 1e-8)
        
        # Set conservative change threshold to prevent excessive changes
        change_threshold = getattr(self.opt, 'max_depth_change_ratio', 0.15)  # Default 15% change threshold
        
        # Apply penalty only to pixels in optimization region with excessive change
        large_change_mask = optimization_mask & (relative_change > change_threshold)
        
        if large_change_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Penalize excessive change, penalty increases as change increases
        excess_change = torch.clamp(relative_change - change_threshold, min=0.0)
        stability_loss = torch.mean(excess_change[large_change_mask] ** 2)
        
        # Final check
        if torch.isnan(stability_loss) or torch.isinf(stability_loss):
            logging.error("Depth stability loss is NaN or infinite")
            return torch.tensor(0.0, device=self.device)
            
        return stability_loss


    def get_cotracker_static_mask(self):
        """
        Generate a binary mask marking Co-tracker static point positions.
        Returns a [T, H, W] boolean tensor, True indicates Co-tracker static point exists at that position.
        Uses caching mechanism to avoid re-computation.
        """
        # Check cache
        if hasattr(self, '_cached_cotracker_mask') and self._cached_cotracker_mask is not None:
            return self._cached_cotracker_mask
        
        T, H, W = self.num_frames, self.shape[0], self.shape[1]
        
        # Initialize an all-False mask
        cotracker_mask = torch.zeros((T, H, W), dtype=torch.bool, device=self.device)
        
        # Get static tracks and visibility
        static_tracks = self.all_tracks_static # (T, N_static, 2)
        static_vis = self.all_vis_static       # (T, N_static)

        # Convert coordinates to integer indices
        coords = static_tracks.long()
        
        for t in range(T):
            # Get visible static points in current frame
            vis_t = static_vis[t]
            if vis_t.sum() == 0:
                continue

            coords_t = coords[t][vis_t] # (N_visible, 2)
            
            # Filter out points outside image boundaries
            valid_mask = (coords_t[:, 0] >= 0) & (coords_t[:, 0] < W) & \
                         (coords_t[:, 1] >= 0) & (coords_t[:, 1] < H)
            
            coords_t_valid = coords_t[valid_mask]
            
            # Mark corresponding positions as True on mask
            cotracker_mask[t, coords_t_valid[:, 1], coords_t_valid[:, 0]] = True
        
        # Cache result
        self._cached_cotracker_mask = cotracker_mask
        
        # Log only on first generation
        logging.info(f"Generated and cached Co-tracker static point mask, total points: {cotracker_mask.sum()}")
        return cotracker_mask





    def visualize_BA_regions(self):
        """
        Visualize BA optimization regions, showing only mask areas without control points.
        """
        import cv2
        import os
        import numpy as np
        from tqdm import tqdm

        # Create save directory
        save_dir = os.path.join(self.opt.output_dir, "BA_regions")
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving BA optimization region visualizations to: {save_dir}")
        
        # Ensure static region mask is created
        if not hasattr(self, 'is_static_strict_tensor') or self.is_static_strict_tensor is None:
            if hasattr(self, 'is_static_strict') and self.is_static_strict is not None:
                self.is_static_strict_tensor = torch.from_numpy(self.is_static_strict).bool().to(self.device)
            else:
                logging.error("Could not find static region mask, cannot visualize.")
                return
                
        # Process each frame
        for f in tqdm(range(min(self.num_frames, self.is_static_strict_tensor.shape[0])), desc="Generating region visualization"):
            if hasattr(self, 'images') and self.images is not None:
                # Get current frame image
                orig_img = self.images[f].cpu().numpy()
                if orig_img.dtype != np.uint8:
                    orig_img = (orig_img * 255).astype(np.uint8)
                
                # Get current frame's static mask
                static_mask = self.is_static_strict_tensor[f].cpu().numpy()
                
                # Create a color mask for display (blue for BA region)
                visualization = orig_img.copy()
                color_mask = np.zeros_like(visualization)
                color_mask[static_mask] = [255, 128, 0]  # Blue-orange color
                
                # Blend images
                alpha = 0.5
                overlay = cv2.addWeighted(visualization, 1.0, color_mask, alpha, 0)
                
                # Save visualization image
                viz_path = os.path.join(save_dir, f"frame_{f:04d}_BA_overlay.png")
                cv2.imwrite(viz_path, overlay)
                
                # Save binary mask image
                mask_img = np.zeros((self.height, self.width), dtype=np.uint8)
                mask_img[static_mask] = 255
                mask_path = os.path.join(save_dir, f"frame_{f:04d}_BA_mask.png")
                cv2.imwrite(mask_path, mask_img)
        
        logging.info(f"Completed BA region mask visualization for all frames.")
        
    def update_static_mask_for_flow(self):
        """
        Update the static mask for flow optimization based on the exclude_epi_from_flow parameter.
        When exclude_epi_from_flow=True, only the inverse of DeVA dynamic regions (value=1) is considered static.
        New feature: Exclude Co-tracker's static points, optimizing only for interpolated static points.
        """
        if not hasattr(self, 'dyn_masks_filters'):
            logging.warning("dyn_masks_filters not found, cannot update static mask for flow.")
            return
        
        # Get shape of all mask arrays
        h, w = self.shape
        num_frames = self.dyn_masks_filters.shape[0]
        
        # Create new mask array
        is_static_for_flow = np.zeros((num_frames, h, w), dtype=bool)
        
        # Set mask based on exclude_epi_from_flow parameter
        if hasattr(self.opt, 'exclude_epi_from_flow') and self.opt.exclude_epi_from_flow:
            # When exclude_epi_from_flow=True, use the inverse of DeVA dynamic regions (value=1) as static
            if hasattr(self, 'dyn_masks_filters_deva') and self.dyn_masks_filters_deva is not None:
                logging.info("exclude_epi_from_flow=True detected, using non-dynamic regions from dyn_masks_filters_deva.")
                
                # Statistics
                static_pixels_total = 0
                total_pixels = num_frames * h * w
                
                for f in range(num_frames):
                    # Inverse of DeVA dynamic regions (value=1) becomes static
                    is_static_for_flow[f] = (self.dyn_masks_filters_deva[f] != 1)
                    
                    # Statistics
                    static_pixels_frame = np.sum(is_static_for_flow[f])
                    static_pixels_total += static_pixels_frame
                    if f == 0:  # Log details for first frame only to avoid spam
                        logging.info(f"Frame {f}: DeVA non-dynamic pixels: {static_pixels_frame}/{h*w} ({static_pixels_frame/(h*w)*100:.2f}%)")
                
                logging.info(f"Average DeVA non-dynamic region ratio over all frames: {static_pixels_total/total_pixels*100:.2f}%")
                logging.info("Using inverse of pure DeVA dynamic regions (unaffected by EPI) for flow optimization.")
            else:
                # If dyn_masks_filters_deva doesn't exist, try to regenerate
                logging.warning("dyn_masks_filters_deva not found! Attempting to regenerate...")
                
                # Call regenerate_deva_masks function to regenerate
                if self.regenerate_deva_masks():
                    logging.info("Successfully regenerated dyn_masks_filters_deva, will proceed to use it for flow optimization.")
                    for f in range(num_frames):
                        is_static_for_flow[f] = (self.dyn_masks_filters_deva[f] != 1)
                else:
                    logging.warning("Failed to regenerate dyn_masks_filters_deva, falling back to using dyn_masks_filters.")
                    for f in range(num_frames):
                        is_static_for_flow[f] = (self.dyn_masks_filters[f] != 1)
                    logging.info("Using non-dynamic regions from dyn_masks_filters for flow optimization.")
        else:
            # Default case: only consider DeVA static regions (value=0) as static
            for f in range(num_frames):
                is_static_for_flow[f] = (self.dyn_masks_filters[f] == 0)
            logging.info("Using default setting: DeVA static regions (label 0) will be used for flow optimization.")
        
        # Check if cotracker points should be used or excluded in flow optimization
        use_cotracker_in_flow = getattr(self.opt, 'use_cotracker_points_in_flow', False)
        
        if use_cotracker_in_flow:
            logging.info("use_cotracker_points_in_flow=True: Co-tracker static points will be INCLUDED in flow optimization.")
        else:
            logging.info("use_cotracker_points_in_flow=False: Co-tracker static points will be EXCLUDED from flow optimization.")
            
            # Get cotracker static point mask and exclude these points
            try:
                cotracker_static_mask = self.get_cotracker_static_mask()
                logging.info("Successfully retrieved Co-tracker static point mask.")
                
                # Statistics before exclusion
                static_pixels_before = np.sum(is_static_for_flow)
                cotracker_static_pixels = cotracker_static_mask.cpu().numpy().sum() if isinstance(cotracker_static_mask, torch.Tensor) else cotracker_static_mask.sum()
                
                # Exclude cotracker's static points from the flow optimization region
                if isinstance(cotracker_static_mask, torch.Tensor):
                    cotracker_static_mask_np = cotracker_static_mask.cpu().numpy()
                else:
                    cotracker_static_mask_np = cotracker_static_mask
                    
                # Ensure shapes match
                if cotracker_static_mask_np.shape == is_static_for_flow.shape:
                    # Exclude cotracker static points from strict static regions to get interpolated static points
                    is_static_for_flow = is_static_for_flow & (~cotracker_static_mask_np)
                    
                    static_pixels_after = np.sum(is_static_for_flow)
                    excluded_pixels = static_pixels_before - static_pixels_after
                    
                    logging.info(f"Pixels before excluding Co-tracker static points: {static_pixels_before}")
                    logging.info(f"Total Co-tracker static points: {cotracker_static_pixels}")
                    logging.info(f"Number of pixels actually excluded: {excluded_pixels}")
                    logging.info(f"Pixels after excluding Co-tracker static points: {static_pixels_after}")
                    logging.info(f"Flow optimization will now run only on interpolated static points, ratio: {static_pixels_after/(num_frames*h*w)*100:.2f}% of total pixels")
                else:
                    logging.warning(f"Co-tracker mask shape {cotracker_static_mask_np.shape} does not match static mask shape {is_static_for_flow.shape}, skipping exclusion.")
                    
            except Exception as e:
                logging.warning(f"Error getting Co-tracker static point mask: {e}")
                logging.warning("Will continue with the original static mask, without excluding Co-tracker points.")
        
        # Count static region pixels
        static_pixels_count = np.sum(is_static_for_flow)
        total_pixels = is_static_for_flow.size
        logging.info(f"Final flow optimization region pixel ratio: {static_pixels_count}/{total_pixels} ({static_pixels_count/total_pixels*100:.2f}%)")
        
        # Update tensor version of the mask (for GPU computation)
        self.is_static_strict_tensor = torch.from_numpy(is_static_for_flow).bool().to(self.device)
        
        # Save a copy for visualization - ensure it's assigned correctly
        self.flow_optimization_mask = is_static_for_flow.copy()
        
        use_cotracker_in_flow = getattr(self.opt, 'use_cotracker_points_in_flow', False)
        
        if hasattr(self.opt, 'exclude_epi_from_flow') and self.opt.exclude_epi_from_flow:
            if use_cotracker_in_flow:
                logging.info("Successfully updated flow optimization region: using inverse of pure DeVA dynamic regions and INCLUDING Co-tracker static points.")
            else:
                logging.info("Successfully updated flow optimization region: using inverse of pure DeVA dynamic regions and EXCLUDING Co-tracker static points.")
        else:
            if use_cotracker_in_flow:
                logging.info("Successfully updated static region mask: excluding EPI regions but INCLUDING Co-tracker static points. Flow optimization on all static points.")
            else:
                logging.info("Successfully updated static region mask: excluding EPI regions and Co-tracker static points. Flow optimization on interpolated points only.")

    def visualize_flow_optimization_regions(self):
        """
        Visualize the region used for flow optimization, showing the mask used when exclude_epi_from_flow=True.
        """
        import cv2
        import os
        import numpy as np
        from tqdm import tqdm

        # Create save directory
        save_dir = os.path.join(self.opt.output_dir, "flow_optimization_regions")
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving flow optimization region visualizations to: {save_dir}")

        # Ensure mask exists
        if not hasattr(self, 'flow_optimization_mask') or self.flow_optimization_mask is None:
            if hasattr(self, 'is_static_strict_tensor') and self.is_static_strict_tensor is not None:
                # Use static region mask as a fallback
                flow_mask = self.is_static_strict_tensor.cpu().numpy()
                logging.info("Using is_static_strict_tensor for flow optimization region visualization.")
            else:
                logging.error("Could not find flow optimization region mask, cannot visualize.")
                return
        else:
            flow_mask = self.flow_optimization_mask
        
        # Check if exclude_epi_from_flow setting is used
        exclude_epi = hasattr(self.opt, 'exclude_epi_from_flow') and self.opt.exclude_epi_from_flow
        
        # Process each frame
        for f in tqdm(range(min(self.num_frames, flow_mask.shape[0])), desc="Generating flow optimization region visualization"):
            # Get current frame's mask
            mask = flow_mask[f].astype(np.uint8) * 255
            
            # Generate visualization images
            # 1. Save original binary mask
            mask_path = os.path.join(save_dir, f"frame_{f:04d}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            # 2. Create RGB visualization (overlay on original image)
            if hasattr(self, 'images') and self.images is not None:
                orig_img = self.images[f].cpu().numpy()
                if orig_img.dtype != np.uint8:
                    orig_img = (orig_img * 255).astype(np.uint8)
                
                # Create color mask (use a bright color for the optimization region)
                color_mask = np.zeros_like(orig_img)
                
                # Use bright yellow to mark the optimization region for better visibility
                color_mask[mask > 0] = [0, 255, 255]  # Yellow
                
                # Calculate percentage of pixels used in optimization
                mask_percentage = np.sum(mask > 0) / mask.size * 100
                
                # Try to get cotracker static point mask for visual comparison
                cotracker_mask_available = False
                cotracker_mask_f = None
                try:
                    cotracker_full_mask = self.get_cotracker_static_mask()
                    if isinstance(cotracker_full_mask, torch.Tensor):
                        cotracker_mask_f = cotracker_full_mask[f].cpu().numpy()
                    else:
                        cotracker_mask_f = cotracker_full_mask[f]
                    cotracker_mask_available = True
                except:
                    pass
                
                # Add text description
                img_with_mask = orig_img.copy()
                title_color = (0, 255, 255)  # Yellow, matching the mask color
                
                cv2.putText(img_with_mask, f"Flow Opt Region: Interpolated Static Points", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                cv2.putText(img_with_mask, f"Strict Static - CoTracker Static", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                cv2.putText(img_with_mask, f"Optimized Area: {mask_percentage:.2f}%", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                
                # If cotracker mask is available, show its stats
                if cotracker_mask_available and cotracker_mask_f is not None:
                    cotracker_pixels = np.sum(cotracker_mask_f)
                    cotracker_percentage = cotracker_pixels / cotracker_mask_f.size * 100
                    cv2.putText(img_with_mask, f"CoTracker Static Area: {cotracker_percentage:.2f}%", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                
                # Create a boundary display to make the region clearer
                mask_dilated = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
                mask_eroded = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
                boundary = mask_dilated - mask_eroded
                
                # Show region boundary on the original image
                boundary_mask = np.zeros_like(orig_img)
                boundary_mask[boundary > 0] = [0, 0, 255]  # Red boundary
                
                # Blend images - first add the colored region, then the boundary
                alpha = 0.5  # Mask transparency
                overlay = cv2.addWeighted(img_with_mask, 1.0, color_mask, alpha, 0)
                # Boundary doesn't need transparency, overlay directly
                overlay[boundary > 0] = boundary_mask[boundary > 0]
                
                # Save overlay image
                overlay_path = os.path.join(save_dir, f"frame_{f:04d}_overlay.png")
                cv2.imwrite(overlay_path, overlay)
                
                # Create a 3-region comparison image: strict static, cotracker static, final optimization region
                if cotracker_mask_available and cotracker_mask_f is not None:
                    # Get original strict static region from is_static_strict_tensor
                    try:
                        if hasattr(self, 'is_static_strict'):
                            strict_static_mask = self.is_static_strict[f]
                        elif hasattr(self, 'dyn_masks_filters'):
                            strict_static_mask = (self.dyn_masks_filters[f] == 0)
                        else:
                            strict_static_mask = np.ones_like(mask, dtype=bool)  # fallback
                        
                        comparison_img = orig_img.copy()
                        
                        # Different colors for different region types
                        # Blue: Strict static region (original)
                        comparison_img[strict_static_mask] = comparison_img[strict_static_mask] * 0.7 + np.array([255, 100, 0]) * 0.3
                        
                        # Red: CoTracker static points
                        comparison_img[cotracker_mask_f] = comparison_img[cotracker_mask_f] * 0.7 + np.array([0, 0, 255]) * 0.3
                        
                        # Yellow: Final flow optimization region (interpolated static points)
                        comparison_img[mask > 0] = comparison_img[mask > 0] * 0.7 + np.array([0, 255, 255]) * 0.3
                        
                        # Add legend
                        cv2.putText(comparison_img, "Blue: Strict Static Region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                        cv2.putText(comparison_img, "Red: CoTracker Static Points", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(comparison_img, "Yellow: Flow Opt Region (Interpolated)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Calculate overlap statistics
                        overlap_pixels = np.sum(strict_static_mask & cotracker_mask_f)
                        remaining_pixels = np.sum(mask > 0)
                        cv2.putText(comparison_img, f"Orig. Static: {np.sum(strict_static_mask)} | CoTracker: {np.sum(cotracker_mask_f)}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(comparison_img, f"Overlap: {overlap_pixels} | Final Opt: {remaining_pixels}", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Save comparison image
                        comparison_path = os.path.join(save_dir, f"frame_{f:04d}_comparison.png")
                        cv2.imwrite(comparison_path, comparison_img)
                        
                    except Exception as e:
                        logging.warning(f"Error creating comparison image: {e}")
                
                # Save another version that highlights the region more
                highlight_mask = np.zeros_like(orig_img)
                # Darken non-optimization region
                highlight_mask[mask == 0] = orig_img[mask == 0] // 2
                # Keep optimization region as is, but with a yellow border
                highlight_mask[mask > 0] = orig_img[mask > 0]
                # Add text description
                cv2.putText(highlight_mask, f"Flow Opt Region: Interpolated Static Points", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                cv2.putText(highlight_mask, f"Strict Static - CoTracker Static", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
                
                # Save highlighted version
                highlight_path = os.path.join(save_dir, f"frame_{f:04d}_highlight.png")
                cv2.imwrite(highlight_path, highlight_mask)
        
        logging.info(f"Completed flow optimization region visualization for all {min(self.num_frames, flow_mask.shape[0])} frames.")
        logging.info("Visualization shows interpolated static points: Strict Static Region minus CoTracker Static Points.")
        logging.info("Generated files include:")
        logging.info("  - frame_XXXX_mask.png: Binary mask")
        logging.info("  - frame_XXXX_overlay.png: Visualization overlayed on the original image")
        logging.info("  - frame_XXXX_highlight.png: Version highlighting the optimization region")
        logging.info("  - frame_XXXX_comparison.png: 3-region comparison (if CoTracker data is available)")
        logging.info(f"Visualizations saved to: {save_dir}")

    def visualize_flow_loss_sampling(self, from_idx, to_idx, original_y_coords, original_x_coords, 
                                   sampled_y_coords, sampled_x_coords, sample_fraction):
        """
        Visualize sampling results in flow consistency loss calculation.
        
        Args:
            from_idx, to_idx: Frame pair indices
            original_y_coords, original_x_coords: All pixel coordinates in original mask
            sampled_y_coords, sampled_x_coords: Sampled pixel coordinates
            sample_fraction: Sampling fraction
        """
        import cv2
        import os
        import numpy as np

        # Create save directory
        save_dir = os.path.join(self.opt.output_dir, "flow_loss_sampling")
        os.makedirs(save_dir, exist_ok=True)

        H, W = self.shape
        total_original = len(original_y_coords)
        total_sampled = len(sampled_y_coords)
        
        # Create visualization masks
        original_mask = torch.zeros((H, W), device=self.device, dtype=torch.bool)
        sampled_mask = torch.zeros((H, W), device=self.device, dtype=torch.bool)
        
        # Mark original pixels and sampled pixels
        original_mask[original_y_coords, original_x_coords] = True
        sampled_mask[sampled_y_coords, sampled_x_coords] = True
        
        # Convert to numpy arrays
        original_mask_np = original_mask.cpu().numpy().astype(np.uint8)
        sampled_mask_np = sampled_mask.cpu().numpy().astype(np.uint8)
        
        # If original images exist, create overlay visualization
        if hasattr(self, 'images') and self.images is not None and from_idx < self.images.shape[0]:
            orig_img = self.images[from_idx].cpu().numpy()
            if orig_img.dtype != np.uint8:
                orig_img = (orig_img * 255).astype(np.uint8)
            
            # Create comparison image: Show original region (blue) and sampled points (red)
            visualization = orig_img.copy()
            
            # Original mask region marked with translucent blue
            blue_overlay = np.zeros_like(orig_img)
            blue_overlay[original_mask_np > 0] = [255, 100, 0]  # Blue
            visualization = cv2.addWeighted(visualization, 0.8, blue_overlay, 0.2, 0)
            
            # Sampled points marked with small red dots
            sampled_coords = np.column_stack(np.where(sampled_mask_np > 0))
            for y, x in sampled_coords:
                cv2.circle(visualization, (x, y), 2, (0, 0, 255), -1)  # Red dot
            
            # Add text info
            title_color = (255, 255, 255)  # White text
            cv2.putText(visualization, f"Uniform Grid Sampling: Frame {from_idx}->{to_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
            cv2.putText(visualization, f"Blue: Original Mask ({total_original} pixels)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(visualization, f"Red: Grid Sampled Points ({total_sampled} pixels)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(visualization, f"Target Fraction: {sample_fraction:.3f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
            cv2.putText(visualization, f"Actual Fraction: {total_sampled/total_original:.3f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
            cv2.putText(visualization, f"Speedup: {total_original/total_sampled:.1f}x", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
            
            # Save visualization image
            viz_path = os.path.join(save_dir, f"flow_loss_sampling_{from_idx:04d}_to_{to_idx:04d}.png")
            cv2.imwrite(viz_path, visualization)
            
            # Create sampling density distribution map
            # Divide image into grids, count sampling density in each grid
            grid_size = 32  # 32x32 grid
            grid_h = H // grid_size
            grid_w = W // grid_size
            
            density_map = np.zeros((grid_size, grid_size))
            original_density_map = np.zeros((grid_size, grid_size))
            
            # Calculate sampling density for each grid
            for y, x in zip(sampled_y_coords.cpu().numpy(), sampled_x_coords.cpu().numpy()):
                grid_y = min(y // grid_h, grid_size - 1)
                grid_x = min(x // grid_w, grid_size - 1)
                density_map[grid_y, grid_x] += 1
            
            for y, x in zip(original_y_coords.cpu().numpy(), original_x_coords.cpu().numpy()):
                grid_y = min(y // grid_h, grid_size - 1)
                grid_x = min(x // grid_w, grid_size - 1)
                original_density_map[grid_y, grid_x] += 1
            
            # Calculate sampling rate map (avoid division by zero)
            sampling_rate_map = np.zeros_like(density_map)
            valid_mask = original_density_map > 0
            sampling_rate_map[valid_mask] = density_map[valid_mask] / original_density_map[valid_mask]
            
            # Resize density map to image size
            density_img = cv2.resize(sampling_rate_map, (W, H), interpolation=cv2.INTER_NEAREST)
            density_colored = cv2.applyColorMap((density_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with original image
            density_overlay = cv2.addWeighted(orig_img, 0.6, density_colored, 0.4, 0)
            
            # Add color bar description
            cv2.putText(density_overlay, f"Grid Sampling Uniformity: Frame {from_idx}->{to_idx}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(density_overlay, "Blue=Sparse Grid, Red=Dense Grid", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(density_overlay, f"Target Rate: {sample_fraction:.3f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save density map
            density_path = os.path.join(save_dir, f"flow_sampling_density_{from_idx:04d}_to_{to_idx:04d}.png")
            cv2.imwrite(density_path, density_overlay)

        # Generate sampling stats file
        stats_path = os.path.join(save_dir, f"sampling_stats_{from_idx:04d}_to_{to_idx:04d}.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Flow consistency loss uniform grid sampling statistics\n")
            f.write(f"Frame pair: {from_idx} -> {to_idx}\n")
            f.write(f"="*40 + "\n\n")
            f.write(f"Sampling method: Uniform grid sampling (ensure uniform spatial distribution)\n")
            f.write(f"Original mask pixels: {total_original}\n")
            f.write(f"Grid sampled pixels: {total_sampled}\n")
            f.write(f"Target sampling fraction: {sample_fraction:.6f}\n")
            f.write(f"Actual sampling fraction: {total_sampled/total_original:.6f}\n")
            f.write(f"Computation reduction: {(1-total_sampled/total_original)*100:.1f}%\n")
            f.write(f"Actual speedup: {total_original/total_sampled:.2f}x\n")
            f.write(f"\nSampling characteristics:\n")
            f.write(f"- Grid uniform distribution, avoids sampling bias\n")
            f.write(f"- Maintains spatial representativeness, covers all regions\n")
            f.write(f"- Suitable for spatial consistency requirements of flow optimization\n")

        logging.info(f"Saving flow loss sampling visualization: Frame {from_idx}->{to_idx}")



    def generate_flow_depth_consistency_report(self):
        """
        Generate flow and depth map consistency report to verify modification effects.
        """
        logging.info("=== Flow and Depth Map Consistency Verification Report ===")
        
        # Basic info
        H, W = self.shape
        logging.info(f"Depth map/Image dimensions: {H}x{W}")
        
        if hasattr(self, 'images') and self.images is not None:
            img_h, img_w = self.images.shape[1:3]
            logging.info(f"Loaded image dimensions: {img_h}x{img_w}")
            
            if img_h != H or img_w != W:
                logging.warning(f"Image dimensions do not match depth map dimensions!")
            else:
                logging.info("✓ Image dimensions match depth map")
        
        # Test size consistency for a few flow predictions
        test_pairs = [(0, 1), (1, 2), (0, 2)] if self.num_frames >= 3 else [(0, 1)]
        
        for from_idx, to_idx in test_pairs:
            if from_idx < self.num_frames and to_idx < self.num_frames:
                try:
                    logging.info(f"\n--- Testing flow pair {from_idx}->{to_idx} ---")
                    
                    # Get raw flow
                    raw_flow = self.get_flow_prediction(from_idx, to_idx)
                    if raw_flow is not None:
                        flow_h, flow_w = raw_flow.shape[:2]
                        logging.info(f"Raw flow size: {flow_h}x{flow_w}")
                        
                        # Validate flow
                        validated_flow = self.validate_flow_depth_consistency(raw_flow, from_idx, to_idx)
                        if validated_flow is not None:
                            final_h, final_w = validated_flow.shape[:2]
                            logging.info(f"Validated flow size: {final_h}x{final_w}")
                            
                            if final_h == H and final_w == W:
                                logging.info("✓ Flow and depth map dimensions match perfectly")
                            else:
                                logging.error("✗ Validated flow size still does not match!")
                                
                            # Check rationality of flow values
                            flow_magnitude = torch.norm(validated_flow, dim=2)
                            max_flow = flow_magnitude.max().item()
                            mean_flow = flow_magnitude.mean().item()
                            median_flow = flow_magnitude.median().item()
                            
                            logging.info(f"Flow statistics: max={max_flow:.2f}, mean={mean_flow:.2f}, median={median_flow:.2f}")
                            
                        else:
                            logging.error("✗ Flow validation failed")
                    else:
                        logging.error("✗ Cannot get flow prediction")
                        
                except Exception as e:
                    logging.error(f"Error testing flow pair {from_idx}->{to_idx}: {e}")
        
        # Check flow model configuration
        if hasattr(self, 'flow_processor'):
            flow_model_type = getattr(self.opt, 'flow_model', 'unknown')
            logging.info(f"\nCurrent flow model: {flow_model_type}")
            
            if flow_model_type == 'unimatch':
                logging.info("✓ Using Unimatch model, high-resolution optimization enabled")
                logging.info("  - Max size limit increased to 1024 pixels")
                logging.info("  - Flow value proportional scaling correction enabled")
                logging.info("  - Detailed size validation logging added")
            else:
                logging.info(f"Using {flow_model_type} model")
        
        logging.info("\n=== Flow Verification System Status ===")
        if hasattr(self, 'validate_flow_depth_consistency'):
            logging.info("✓ Flow-depth consistency validation function loaded")
        if hasattr(self, 'get_flow_prediction'):
            logging.info("✓ Flow prediction function size validation enabled")
        
        logging.info("\n=== Verification Report Completed ===")
        
        return True
    
    def create_init_static_points(self):
        """
        Create co-tracker static points specifically for init stage, excluding EPI+dynamic regions.
        """
        logging.info("Creating init-specific static co-tracker points (excluding EPI+dynamic regions)...")
        
        # First check if necessary attributes exist
        required_attrs = ['all_labels', 'all_tracks_static', 'all_vis_static', 'track_init_frames', 'confidences']
        for attr in required_attrs:
            if not hasattr(self, attr):
                logging.error(f"Required attribute {attr} not found in create_init_static_points")
                return
                
        # Add debug info
        logging.info(f"all_labels shape: {self.all_labels.shape}")
        logging.info(f"all_tracks_static shape: {self.all_tracks_static.shape}")
        logging.info(f"all_vis_static shape: {self.all_vis_static.shape}")
        
        # Get all co-tracker points marked as static
        static_point_mask = (self.all_labels == 0)
        logging.info(f"Static point mask sum: {static_point_mask.sum().item()}")
        
        if static_point_mask.sum() == 0:
            logging.warning("No static co-tracker points found for init stage")
            # Create empty init static point set
            self.all_tracks_static_init_filtered = torch.zeros((self.num_frames, 0, 2), device=self.device)
            self.all_vis_static_init_filtered = torch.zeros((self.num_frames, 0), dtype=torch.bool, device=self.device)
            self.all_tracks_static_init_frames_filtered = torch.zeros(0, dtype=torch.int, device=self.device)
            self.static_confidences_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
            self.num_points_static_init_filtered = 0
            return
        
        # Get indices of static points
        static_indices = torch.where(static_point_mask)[0]
        
        # Check if mask is loaded
        if not hasattr(self, 'dyn_masks_filters') or self.dyn_masks_filters is None:
            logging.warning("dyn_masks_filters not loaded yet, using all static co-tracker points for init")
            # If mask not loaded yet, use all static points
            valid_static_indices = static_indices
        else:
            # Filter out static points falling in EPI dynamic regions (label 3) and DeVA dynamic regions (label 1)
            valid_static_indices = []
            
            for idx in static_indices:
                point_valid = True
                point_in_valid_region_count = 0
                point_total_vis_count = 0
                
                # Check if the point falls in valid region in all visible frames
                for f in range(self.num_frames):
                    try:
                        if f < self.all_vis.shape[0] and idx < self.all_vis.shape[1] and self.all_vis[f, idx]:  # If point is visible in this frame
                            point_total_vis_count += 1
                            
                            # Get pixel coordinates of the point
                            if f < self.all_tracks.shape[0] and idx < self.all_tracks.shape[1]:
                                x, y = self.all_tracks[f, idx].long()
                                
                                # Ensure coordinates are within image range and reasonable range
                                if (0 <= x < self.shape[1] and 0 <= y < self.shape[0] and 
                                    x >= 0 and y >= 0 and x < 10000 and y < 10000):  # Extra range check
                                    # Check if mask exists and dimensions are correct
                                    if (hasattr(self, 'dyn_masks_filters') and 
                                        self.dyn_masks_filters is not None and
                                        f < self.dyn_masks_filters.shape[0] and
                                        y < self.dyn_masks_filters.shape[1] and 
                                        x < self.dyn_masks_filters.shape[2]):
                                        # Get mask label at this position
                                        mask_label = self.dyn_masks_filters[f, y, x]
                                        
                                        # Check if in valid region (exclude label 1=DeVA dynamic and label 3=EPI dynamic)
                                        if mask_label not in [1, 3]:  # Allow label 0=static and label 2=boundary
                                            point_in_valid_region_count += 1
                    except Exception as e:
                        logging.warning(f"Error processing point {idx} at frame {f}: {e}")
                        continue
                
                # Keep the point if it is in valid region in most visible frames
                if point_total_vis_count > 0:
                    valid_ratio = point_in_valid_region_count / point_total_vis_count
                    # Require at least 80% of visible frames to be in valid region
                    if valid_ratio >= 0.8:
                        valid_static_indices.append(idx)
            
            valid_static_indices = torch.tensor(valid_static_indices, device=self.device)
        
        # Create static point data specific for init
        if len(valid_static_indices) > 0:
            try:
                # Ensure indices are within valid range
                max_static_idx = self.all_tracks_static.shape[1] - 1
                valid_static_indices = valid_static_indices[valid_static_indices <= max_static_idx]
                
                if len(valid_static_indices) > 0:
                    self.all_tracks_static_init_filtered = self.all_tracks_static[:, valid_static_indices, :]
                    self.all_vis_static_init_filtered = self.all_vis_static[:, valid_static_indices]
                    self.all_tracks_static_init_frames_filtered = self.track_init_frames[valid_static_indices]
                    self.static_confidences_init_filtered = self.static_confidences[:, valid_static_indices]
                    self.num_points_static_init_filtered = valid_static_indices.shape[0]
                    
                    # --- New: Create filtered depth information ---
                    if hasattr(self, 'all_tracks_static_depth') and self.all_tracks_static_depth is not None:
                        self.all_tracks_static_depth_init_filtered = self.all_tracks_static_depth[:, valid_static_indices]
                        logging.info("Created filtered depth information for init static points")
                    else:
                        logging.warning("all_tracks_static_depth not available, will be created later")
                        self.all_tracks_static_depth_init_filtered = None
                else:
                    logging.warning("All valid static indices are out of range")
                    # Create empty set
                    self.all_tracks_static_init_filtered = torch.zeros((self.num_frames, 0, 2), device=self.device)
                    self.all_vis_static_init_filtered = torch.zeros((self.num_frames, 0), dtype=torch.bool, device=self.device)
                    self.all_tracks_static_init_frames_filtered = torch.zeros(0, dtype=torch.int, device=self.device)
                    self.static_confidences_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
                    self.all_tracks_static_depth_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
                    self.num_points_static_init_filtered = 0
                    
            except Exception as e:
                logging.error(f"Error creating filtered static points data: {e}")
                # Create empty set as fallback
                self.all_tracks_static_init_filtered = torch.zeros((self.num_frames, 0, 2), device=self.device)
                self.all_vis_static_init_filtered = torch.zeros((self.num_frames, 0), dtype=torch.bool, device=self.device)
                self.all_tracks_static_init_frames_filtered = torch.zeros(0, dtype=torch.int, device=self.device)
                self.static_confidences_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
                self.all_tracks_static_depth_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
                self.num_points_static_init_filtered = 0
                return
            
            logging.info(f"Created init-specific static points: {self.num_points_static_init_filtered} points")
            logging.info(f"Original static points: {self.num_points_static}")
            logging.info(f"Filtered ratio: {self.num_points_static_init_filtered/self.num_points_static*100:.1f}%")
            
            # Statistics - Add safety check
            try:
                if hasattr(self, 'all_vis_static') and self.all_vis_static is not None and self.all_vis_static.numel() > 0:
                    total_vis_original = self.all_vis_static.sum().item()
                else:
                    total_vis_original = 0
                    logging.warning("all_vis_static is empty or None")
                    
                if hasattr(self, 'all_vis_static_init_filtered') and self.all_vis_static_init_filtered is not None and self.all_vis_static_init_filtered.numel() > 0:
                    total_vis_filtered = self.all_vis_static_init_filtered.sum().item()
                else:
                    total_vis_filtered = 0
                    logging.warning("all_vis_static_init_filtered is empty or None")
                    
                logging.info(f"Total visible points - Original: {total_vis_original}, Filtered: {total_vis_filtered}")
            except Exception as e:
                logging.warning(f"Error computing visibility statistics: {e}")
                logging.info("Skipping visibility statistics due to error")
            
        else:
            logging.warning("No valid static co-tracker points found after filtering EPI+dynamic regions")
            # Create empty init static point set
            self.all_tracks_static_init_filtered = torch.zeros((self.num_frames, 0, 2), device=self.device)
            self.all_vis_static_init_filtered = torch.zeros((self.num_frames, 0), dtype=torch.bool, device=self.device)
            self.all_tracks_static_init_frames_filtered = torch.zeros(0, dtype=torch.int, device=self.device)
            self.static_confidences_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
            self.all_tracks_static_depth_init_filtered = torch.zeros((self.num_frames, 0), device=self.device)
            self.num_points_static_init_filtered = 0


        
    def analyze_depth_changes(self, depth_before, depth_after):
        """
        Analyze depth map changes
        """
        logging.info("=== Analyzing Depth Map Changes ===")
        
        # Calculate changes
        depth_diff = depth_after - depth_before
        depth_ratio = depth_after / (depth_before + 1e-8)
        
        # Statistics
        logging.info(f"Depth change statistics:")
        logging.info(f"  Mean change: {depth_diff.mean():.4f}")
        logging.info(f"  Change std dev: {depth_diff.std():.4f}")
        logging.info(f"  Max change: {depth_diff.max():.4f}")
        logging.info(f"  Min change: {depth_diff.min():.4f}")
        
        logging.info(f"Depth ratio statistics:")
        logging.info(f"  Mean ratio: {depth_ratio.mean():.4f}")
        logging.info(f"  Ratio std dev: {depth_ratio.std():.4f}")
        logging.info(f"  Max ratio: {depth_ratio.max():.4f}")
        logging.info(f"  Min ratio: {depth_ratio.min():.4f}")
        
        # Analyze change distribution
        large_changes = (torch.abs(depth_diff) > 1.0).float().mean().item()
        small_changes = (torch.abs(depth_diff) < 0.1).float().mean().item()
        logging.info(f"  Large change (>1.0) ratio: {large_changes:.4f}")
        logging.info(f"  Small change (<0.1) ratio: {small_changes:.4f}")
        
        # Save change visualization
        change_save_dir = os.path.join(self.opt.output_dir, "depth_changes")
        os.makedirs(change_save_dir, exist_ok=True)
        
        for f in range(min(5, self.num_frames)):
            # Save change map
            diff_frame = depth_diff[f].detach().cpu().numpy()
            diff_normalized = np.clip((diff_frame + 2) / 4, 0, 1)  # Normalize to 0-1
            diff_uint8 = (diff_normalized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(change_save_dir, f"frame_{f:04d}_depth_change.png"), diff_uint8)
            
            # Save ratio map
            ratio_frame = depth_ratio[f].detach().cpu().numpy()
            ratio_normalized = np.clip(ratio_frame, 0.5, 2.0)  # Limit to 0.5-2.0 range
            ratio_uint8 = ((ratio_normalized - 0.5) / 1.5 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(change_save_dir, f"frame_{f:04d}_depth_ratio.png"), ratio_uint8)
        
        logging.info("=== Depth Map Change Analysis Completed ===")
        
    def check_flow_optimization_parameters(self):
        """
        Check key parameters for flow optimization.
        """
        logging.info("=== Checking Flow Optimization Parameters ===")
        
        # Check key parameters
        params_to_check = [
            'flow_weight', 'ba_loss_weight', 'flow_lr', 'num_flow_epochs',
            'flow_opt_depth', 'flow_opt_pose', 'flow_opt_intrinsics',
            'use_weighted_flow', 'flow_loss_sample_fraction',
            'depth_preservation_weight', 'depth_smoothness_weight', 'depth_stability_weight'
        ]
        
        for param in params_to_check:
            value = getattr(self.opt, param, None)
            if value is not None:
                logging.info(f"  {param}: {value}")
            else:
                logging.warning(f"  {param}: Not set")
        
        # Check optimizer status
        if hasattr(self, 'optimizers'):
            for name, optimizer in self.optimizers.items():
                if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    lr = optimizer.param_groups[0]['lr']
                    param_count = sum(p.numel() for p in optimizer.param_groups[0]['params'])
                    logging.info(f"  Optimizer {name}: lr={lr}, param_count={param_count}")
        
        logging.info("=== Parameter Check Completed ===")

    def compute_masked_flow_loss(self, pairs, flow_predictions, optimization_mask):
        """
        Flow consistency weighted loss calculation function.
        No longer uses sampling, but calculates flow consistency for all optimization region points,
        using consistency loss as weight to calculate final depth optimization loss.
        
        Args:
            pairs: List of frame pairs
            flow_predictions: Flow prediction dictionary
            optimization_mask: Optimization mask
        
        Returns:
            Weighted flow loss
        """
        if len(pairs) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        total_weighted_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_weight = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Ensure optimization_mask is on correct device
        if isinstance(optimization_mask, torch.Tensor):
            optimization_mask = optimization_mask.to(self.device)
        elif isinstance(optimization_mask, np.ndarray):
            optimization_mask = torch.from_numpy(optimization_mask).to(self.device)
        
        for from_idx, to_idx in pairs:
            # Get optimization mask for this frame
            frame_mask = optimization_mask[from_idx].to(self.device)
            if frame_mask.sum() == 0:
                continue
                
            # Get flow prediction
            flow_pred = flow_predictions.get((from_idx, to_idx))
            if flow_pred is None:
                continue
            
            flow_pred = flow_pred.to(self.device)
                
            # Calculate flow consistency weighted loss
            weighted_loss, weight = self.compute_flow_weighted_loss(from_idx, to_idx, flow_pred, frame_mask)
            
            total_weighted_loss = total_weighted_loss + weighted_loss
            total_weight = total_weight + weight
        
        if total_weight > 0:
            return total_weighted_loss / total_weight
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_flow_weighted_loss(self, from_idx, to_idx, flow_pred, mask):
        """
        Compute pure flow consistency weighted loss.
        Use flow consistency between frames as weight, correct geometry using flow.
        
        Args:
            from_idx, to_idx: Frame indices
            flow_pred: Flow prediction (H, W, 2)
            mask: Optimization mask (H, W)
        
        Returns:
            weighted_loss: Weighted loss
            total_weight: Total weight
        """
        H, W = mask.shape
        device = self.device
        
        # Get pixel coordinates of mask region
        y_coords, x_coords = torch.where(mask > 0)
        if len(y_coords) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Compute pure flow consistency weights (without using any geometric info)
        # Points with good flow consistency across frames get high weights
        flow_weights = self.compute_pure_flow_consistency_weights(
            y_coords, x_coords, flow_pred
        )
        
        # Get geometric parameters for loss calculation (make geometry conform to flow)
        depth_from = self.get_depth(from_idx).to(device)  # (H, W)
        depth_to = self.get_depth(to_idx).to(device)     # (H, W)
        
        # Get camera parameters
        R_from, t_from = self.get_poses(from_idx)
        R_to, t_to = self.get_poses(to_idx)
        K = self.get_intrinsics_K([from_idx])[0].to(device)
        
        # Move tensors to correct device
        R_from, t_from = R_from.to(device), t_from.to(device)
        R_to, t_to = R_to.to(device), t_to.to(device)
        
        # Compute depth reprojection loss (make geometric flow close to flow prediction)
        # Assuming flow is accurate, use flow to correct geometry
        depth_loss = self.compute_depth_reprojection_loss(
            y_coords, x_coords, depth_from, depth_to,
            R_from, t_from, R_to, t_to, K, flow_pred
        )
        
        # Weight geometric loss using pure flow consistency weights
        # Points with good flow consistency (large weight) contribute more to geometric correction
        weighted_loss = (depth_loss * flow_weights).sum()
        total_weight = flow_weights.sum()
        
        return weighted_loss, total_weight

    def compute_pure_flow_consistency_weights(self, y_coords, x_coords, flow_pred):
        """
        Compute weights based purely on flow consistency across frames.
        Evaluate flow quality through cycle consistency of forward and backward flow.
        Does not use any geometric info (depth, pose, etc.).
        
        Args:
            y_coords, x_coords: Pixel coordinates (N,)
            flow_pred: Forward flow prediction (H, W, 2) - from->to
        
        Returns:
            weights: Weight for each point (N,) - Better flow consistency means higher weight
        """
        device = self.device
        H, W = flow_pred.shape[:2]
        
        # Get forward flow values at specified points
        flow_forward = flow_pred[y_coords, x_coords]  # (N, 2) - from->to flow
        
        # Calculate position after forward warp
        original_coords = torch.stack([x_coords.float(), y_coords.float()], dim=1).to(device)  # (N, 2)
        forward_coords = original_coords + flow_forward  # (N, 2)
        
        # Get backward flow (to->from)
        # Approximate backward flow by sampling forward flow at forward_coords and negating it
        flow_backward = self.get_backward_flow(flow_pred, forward_coords, H, W)  # (N, 2)
        
        # Calculate position after backward warp (back to original)
        backward_coords = forward_coords + flow_backward  # (N, 2)
        
        # Calculate cycle consistency error - difference between original position and warped back position
        cycle_error = torch.norm(backward_coords - original_coords, dim=1)  # (N,)
        
        # Polarize weight distribution: give strong weights to consistent points, near 0 to poor points
        # Read parameters from config
        sigma = getattr(self.opt, 'cycle_consistency_sigma', 1.0)  
        min_weight = getattr(self.opt, 'min_cycle_weight', 0.001)   # Lower minimum weight
        max_error = getattr(self.opt, 'max_cycle_error', 5.0)      
        quality_threshold = getattr(self.opt, 'flow_quality_threshold', 2.0)  # Quality threshold
        
        # Clamp excessive errors
        cycle_error = torch.clamp(cycle_error, 0, max_error)
        
        # [Phase 1] Basic Gaussian weight calculation
        base_weights = torch.exp(-cycle_error**2 / (2 * sigma**2))
        
        # [Phase 2] Polarization processing - make weight distribution more extreme
        # 1. Quality threshold filtering: significantly reduce weight for points exceeding threshold
        quality_mask = cycle_error <= quality_threshold  # High quality points
        
        # 2. Use exponential enhancement for high quality points
        enhanced_weights = base_weights.clone()
        
        # High quality points: further enhance weight (power function increases difference)
        power_factor = 3.0  # Exponential factor, larger means more difference
        enhanced_weights[quality_mask] = enhanced_weights[quality_mask] ** (1.0 / power_factor)
        
        # Low quality points: significantly reduce weight
        penalty_factor = 4.0  # Penalty factor
        enhanced_weights[~quality_mask] = enhanced_weights[~quality_mask] ** penalty_factor
        
        # 3. Re-polarization: Use sigmoid-like function to further widen difference
        # Map weights to a more extreme distribution
        sharp_factor = 10.0  # Sharpening factor, larger means more extreme distribution
        center_point = 0.5   # Center point
        
        # Sigmoid sharpening: f(x) = 1 / (1 + exp(-sharp_factor * (x - center_point)))
        sharpened_weights = 1.0 / (1.0 + torch.exp(-sharp_factor * (enhanced_weights - center_point)))
        
        # 4. Final weights = Sharpened weights + Minimum weight
        final_weights = sharpened_weights + min_weight
        
        # 5. Bonus for excellent points
        excellent_mask = cycle_error <= (quality_threshold * 0.5)  # Exceptionally good points
        bonus_factor = 2.0  # Bonus factor
        final_weights[excellent_mask] *= bonus_factor
        
        # Log weight distribution stats
        logging.info(f"🎯 Weight distribution stats:")
        logging.info(f"  Error range: min={cycle_error.min().item():.3f}, max={cycle_error.max().item():.3f}")
        logging.info(f"  Weight range: min={final_weights.min().item():.3f}, max={final_weights.max().item():.3f}")
        logging.info(f"  High quality point ratio: {quality_mask.float().mean().item():.3f}")
        logging.info(f"  Excellent point ratio: {excellent_mask.float().mean().item():.3f}")
        
        weights = final_weights
        
        return weights
    
    
    def get_backward_flow(self, flow_forward, target_coords, H, W):
        """
        Simplified backward flow calculation.
        Obtain backward flow estimate through simple interpolation.
        
        Args:
            flow_forward: Forward flow (H, W, 2)
            target_coords: Target coordinates (N, 2)
            H, W: Image dimensions
            
        Returns:
            backward_flow: Backward flow values (N, 2)
        """
        device = flow_forward.device
        
        # Ensure coordinates are within valid range
        valid_mask = (target_coords[:, 0] >= 0) & (target_coords[:, 0] < W) & \
                     (target_coords[:, 1] >= 0) & (target_coords[:, 1] < H)
        
        # Initialize backward flow
        backward_flow = torch.zeros_like(target_coords)
        
        if valid_mask.sum() > 0:
            valid_coords = target_coords[valid_mask]
            
            # Simplified bilinear interpolation to get flow values
            x_coords = torch.clamp(valid_coords[:, 0], 0, W - 1)
            y_coords = torch.clamp(valid_coords[:, 1], 0, H - 1)
            
            # Get nearest neighbor flow values (simplified version, avoiding complex interpolation)
            x_int = torch.round(x_coords).long()
            y_int = torch.round(y_coords).long()
            
            # Ensure indices are within valid range
            x_int = torch.clamp(x_int, 0, W - 1)
            y_int = torch.clamp(y_int, 0, H - 1)
            
            # Get flow values and negate for backward flow
            interpolated_flow = flow_forward[y_int, x_int]  # (N, 2)
            backward_flow[valid_mask] = -interpolated_flow
        
        # For invalid coordinates, backward flow is zero (will lead to large cycle error, low weight)
        return backward_flow

    def compute_depth_reprojection_loss(self, y_coords, x_coords, depth_from, depth_to,
                                      R_from, t_from, R_to, t_to, K, flow_pred):
        """
        Compute simplified flow reprojection loss.
        Directly compare 2D difference between geometric projection flow and predicted flow, avoiding complex depth interpolation.
        
        Args:
            y_coords, x_coords: Pixel coordinates
            depth_from, depth_to: Depth map (only depth_from is used here)
            R_from, t_from, R_to, t_to: Camera poses
            K: Intrinsic matrix
            flow_pred: Flow prediction
        
        Returns:
            losses: Flow consistency loss for each point (N,)
        """
        logging.info(f"🚀 Starting execution of compute_depth_reprojection_loss, Processing {len(y_coords)} points")
        device = self.device
        
        # Ensure coordinates are tensors and on correct device
        if not isinstance(y_coords, torch.Tensor):
            y_coords = torch.tensor(y_coords)
        if not isinstance(x_coords, torch.Tensor):
            x_coords = torch.tensor(x_coords)
        
        y_coords = y_coords.to(device)
        x_coords = x_coords.to(device)
        
        # Get depth of points in current frame
        depths_from = depth_from[y_coords, x_coords]  # (N,)
        depths_from = depths_from.to(device)
        
        # Convert 2D points to 3D points (camera coordinate system)
        points_2d_homo = torch.stack([
            x_coords.float(), y_coords.float(), torch.ones_like(x_coords.float())
        ], dim=1).to(device)  # (N, 3)
        
        K_inv = torch.inverse(K.to(device))
        points_cam = (K_inv @ points_2d_homo.T).T  # (N, 3)
        points_3d_from = points_cam * depths_from.unsqueeze(1)  # (N, 3)
        
        # Project to 'to' frame, get geometric flow
        # Transform from 'from' frame camera coordinate system to world coordinate system
        # points_3d_from: (N, 3), R_from: (3, 3), t_from: (3,)
        
        # Ensure R_from and t_from shapes
        R_from = R_from.to(device)
        t_from = t_from.to(device)
        R_to = R_to.to(device) 
        t_to = t_to.to(device)
        
        # Ensure tensor shapes are correct - safer processing method
        
        # Ensure rotation matrix is (3, 3)
        if R_from.dim() > 2:
            R_from = R_from.squeeze()
        if R_from.shape != (3, 3):
            raise ValueError(f"R_from shape should be (3, 3), got {R_from.shape}")
            
        if R_to.dim() > 2:
            R_to = R_to.squeeze()
        if R_to.shape != (3, 3):
            raise ValueError(f"R_to shape should be (3, 3), got {R_to.shape}")
        
        # Ensure translation vector is (3,)
        if t_from.dim() > 1:
            t_from = t_from.squeeze()
        if t_from.shape != (3,):
            raise ValueError(f"t_from shape should be (3,), got {t_from.shape}")
            
        if t_to.dim() > 1:
            t_to = t_to.squeeze()
        if t_to.shape != (3,):
            raise ValueError(f"t_to shape should be (3,), got {t_to.shape}")
        
        # Ensure 3D points shape is (N, 3)
        if points_3d_from.shape[1] != 3:
            raise ValueError(f"points_3d_from should have shape (N, 3), got {points_3d_from.shape}")
        
        # Transform: P_world = R_from^T * (P_cam - t_from)
        # Using matrix multiplication: (N, 3) @ (3, 3) = (N, 3)
        points_world = (points_3d_from - t_from.unsqueeze(0)) @ R_from.T  # (N, 3)
        
        # Transform from world coordinate system to 'to' frame camera coordinate system
        # P_cam_to = R_to * P_world + t_to
        points_to_cam = points_world @ R_to.T + t_to.unsqueeze(0)  # (N, 3)
        
        # Project to 'to' frame image plane
        # points_to_cam: (N, 3), K: (3, 3)
        points_to_2d_homo = points_to_cam @ K.T  # (N, 3)
        
        # Ensure homogeneous coordinate 3rd component is not zero to avoid division by zero
        z_coords = points_to_2d_homo[:, 2]  # (N,)
        z_coords = torch.clamp(z_coords, min=1e-8)  # Avoid division by zero
        
        # Perform perspective division to get 2D coordinates
        points_to_2d = points_to_2d_homo[:, :2] / z_coords.unsqueeze(1)  # (N, 2)
        
        # Calculate geometric flow (theoretical flow based on camera pose and depth)
        original_coords = torch.stack([x_coords.float(), y_coords.float()], dim=1).to(device)  # (N, 2)
        
        # Ensure both tensors are shape (N, 2)
        assert points_to_2d.shape == original_coords.shape, f"Shape mismatch: points_to_2d {points_to_2d.shape}, original_coords {original_coords.shape}"
        
        # Fix coordinate system inconsistency:
        # Adjust geometric flow direction to match predicted flow (from_frame → to_frame)
        geometric_flow_raw = points_to_2d - original_coords  # (N, 2)
        
        # Get predicted flow first for coordinate system matching
        predicted_flow_temp = flow_pred[y_coords, x_coords].to(device)  # (N, 2)
        
        # Analyze direction distribution of both flows
        geo_x_negative_ratio = (geometric_flow_raw[:, 0] < 0).float().mean().item()
        pred_x_positive_ratio = (predicted_flow_temp[:, 0] > 0).float().mean().item()
        
        geometric_flow = geometric_flow_raw.clone()
        
        # Automatically detect and fix coordinate system inconsistency
        if geo_x_negative_ratio > 0.8 and pred_x_positive_ratio > 0.7:
            logging.info(f"🔄 Detected X-direction inconsistency (Geo negative ratio: {geo_x_negative_ratio:.3f}, Pred positive ratio: {pred_x_positive_ratio:.3f})")
            geometric_flow[:, 0] = -geometric_flow[:, 0]  # Flip X direction
            logging.info("🔄 Flipped geometric flow X direction")
        
        # Also check Y direction consistency (though Y direction issues are rare)
        geo_y_negative_ratio = (geometric_flow_raw[:, 1] < 0).float().mean().item()
        pred_y_positive_ratio = (predicted_flow_temp[:, 1] > 0).float().mean().item()
        
        if geo_y_negative_ratio > 0.8 and pred_y_positive_ratio > 0.7:
            logging.info(f"🔄 Detected Y-direction inconsistency (Geo negative ratio: {geo_y_negative_ratio:.3f}, Pred positive ratio: {pred_y_positive_ratio:.3f})")
            geometric_flow[:, 1] = -geometric_flow[:, 1]  # Flip Y direction
            logging.info("🔄 Flipped geometric flow Y direction")
        
        # Use previously acquired predicted flow
        predicted_flow = predicted_flow_temp  # (N, 2) already on correct device
        
        # Add debug info to confirm units and range of both flows
        geo_mag = torch.norm(geometric_flow, dim=1)
        pred_mag = torch.norm(predicted_flow, dim=1)
        
        logging.info(f"🔍 Geometric flow stats: max={geo_mag.max().item():.2f}, mean={geo_mag.mean().item():.2f}")
        logging.info(f"🔍 Predicted flow stats: max={pred_mag.max().item():.2f}, mean={pred_mag.mean().item():.2f}")
        
        # Confirm coordinate system consistency: check xy order of flow
        logging.info(f"🔍 Geometric flow first 5 points: {geometric_flow[:5]}")
        logging.info(f"🔍 Predicted flow first 5 points: {predicted_flow[:5]}")
        
        # Analyze flow direction distribution
        geo_x_positive = (geometric_flow[:, 0] > 0).float().mean().item()
        geo_y_positive = (geometric_flow[:, 1] > 0).float().mean().item()
        pred_x_positive = (predicted_flow[:, 0] > 0).float().mean().item() 
        pred_y_positive = (predicted_flow[:, 1] > 0).float().mean().item()
        
        logging.info(f"🔍 Geometric flow direction distribution: X positive ratio={geo_x_positive:.3f}, Y positive ratio={geo_y_positive:.3f}")
        logging.info(f"🔍 Predicted flow direction distribution: X positive ratio={pred_x_positive:.3f}, Y positive ratio={pred_y_positive:.3f}")
        
        # Check if coordinate system transform is needed
        if geo_x_positive < 0.3 and pred_x_positive > 0.7:
            logging.warning("⚠️  Geometric flow and predicted flow X directions are opposite, coordinate system transform might be needed")
        if geo_y_positive < 0.3 and pred_y_positive > 0.7:
            logging.warning("⚠️  Geometric flow and predicted flow Y directions are opposite, coordinate system transform might be needed")
        
        # Calculate difference between geometric flow and predicted flow (L2 distance)
        flow_diff = torch.norm(geometric_flow - predicted_flow, dim=1)  # (N,)
        
        return flow_diff


    def load_unimatch_model(self):
        """Load UniMatch model, using exactly same parameters as inference_flow.py"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use exactly same parameters as inference_flow.py
        model = UniMatch(feature_channels=128,
                        num_scales=2,
                        upsample_factor=4,
                        num_head=1,
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        reg_refine=True,
                        task='flow').to(device)
        
        # Load pre-trained weights
        checkpoint_path = "./dynamicBA/unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        print(f'Loading UniMatch checkpoint from {checkpoint_path}')
        loc = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        
        return model

    def load_flow_model(self):
        """Load flow model - compatible with existing interface"""
        try:
            logging.info("Processing loading UniMatch flow model...")
            self.flow_processor = self.load_unimatch_model()
            logging.info("✅ UniMatch flow model loaded successfully")
        except Exception as e:
            logging.error(f"❌ Flow model loading failed: {e}")
            self.flow_processor = None
            raise

    def get_flow_prediction(self, from_idx, to_idx):
        """Get flow prediction between two frames"""
        if not hasattr(self, 'flow_processor') or self.flow_processor is None:
            logging.warning("Flow processor not loaded, cannot predict flow")
            return None
            
        try:
            # Get source and target frame images
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get images from self.images (Correction: use self.images instead of self.imgs)
            img_from = self.images[from_idx].float().to(device)  # [H, W, 3]
            img_to = self.images[to_idx].float().to(device)     # [H, W, 3]
            
            # Convert to [1, 3, H, W] format
            img_from = img_from.permute(2, 0, 1).unsqueeze(0)
            img_to = img_to.permute(2, 0, 1).unsqueeze(0)
            
            # Use UniMatch to predict flow
            with torch.no_grad():
                flow_pred = self.predict_flow_with_unimatch(
                    self.flow_processor, 
                    img_from, 
                    img_to, 
                    device
                )
                
                # Convert to [H, W, 2] format
                flow = flow_pred[0].permute(1, 2, 0)  # [H, W, 2]
                
            return flow
            
        except Exception as e:
            logging.error(f"Flow prediction failed (Frame {from_idx}->{to_idx}): {e}")
            return None

    def predict_flow_with_unimatch(self, model, image1, image2, device):
        """Use UniMatch to predict flow, parameters exactly matching inference_flow.py"""
        # Exactly same parameters as inference_flow.py
        attn_type = 'swin'
        attn_splits_list = [2, 8]
        corr_radius_list = [-1, 4]
        prop_radius_list = [-1, 1]
        num_reg_refine = 6
        padding_factor = 32
        
        # Ensure images are float type
        if image1.dtype == torch.uint8:
            image1 = image1.float()
        if image2.dtype == torch.uint8:
            image2 = image2.float()
            
        # Image preprocessing - keep consistent with inference_flow.py
        transpose_img = False
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        # Calculate padding size
        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                       int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
        ori_size = image1.shape[-2:]
        
        # Resize images if needed
        if nearest_size[0] != ori_size[0] or nearest_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=nearest_size, mode='bilinear', align_corners=True)
            image2 = F.interpolate(image2, size=nearest_size, mode='bilinear', align_corners=True)

        # Predict flow
        results_dict = model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=False)

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # Resize back to original size
        if nearest_size[0] != ori_size[0] or nearest_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear', align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / nearest_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / nearest_size[-2]

        # If transposed before, transpose back
        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        return flow_pr

   


    def optimize_simple_flow(self, pairs, flow_predictions):
        """
        Simplified flow optimization: Keep only core functionality
        """
        # Create optimization mask
        interpolated_mask, _ = self.create_interpolated_static_optimization_mask()
        
        if interpolated_mask.sum().item() == 0:
            return
        
        if not hasattr(self, 'reproj_depths_param'):
            return
        
        # Set parameters
        max_epochs = getattr(self.opt, 'num_flow_epochs', 50)
        flow_weight = getattr(self.opt, 'flow_weight', 0.1)
        flow_lr = getattr(self.opt, 'flow_lr', 1e-4)
        
        if "reproj_depth" in self.optimizers:
            for param_group in self.optimizers["reproj_depth"].param_groups:
                param_group['lr'] = flow_lr
        
        if "reproj_depth" not in self.optimizers:
            return
        if "reproj_depth" not in self.active_optimizers:
            self.active_optimizers.append("reproj_depth")
        
        for epoch in range(max_epochs):
            if "reproj_depth" in self.optimizers:
                self.optimizers["reproj_depth"].zero_grad()
            
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            valid_pairs = 0
            
            for from_idx, to_idx in pairs:
                pair_key = f"{from_idx}_{to_idx}"
                if pair_key not in flow_predictions:
                    continue
                
                flow_pred = flow_predictions[pair_key]
                mask = interpolated_mask[from_idx]
                
                if mask.sum().item() == 0:
                    continue
                
                # Get pixel coordinates within mask
                y_coords, x_coords = torch.where(mask)
                if len(y_coords) == 0:
                    continue
                
                # Sample to improve efficiency
                sample_ratio = getattr(self.opt, 'flow_sample_ratio', 0.1)
                if sample_ratio < 1.0:
                    num_samples = int(len(y_coords) * sample_ratio)
                    if num_samples > 0:
                        indices = torch.randperm(len(y_coords))[:num_samples]
                        y_coords = y_coords[indices]
                        x_coords = x_coords[indices]
                    else:
                        continue
                
                try:
                    # Check depth parameter shape and coordinate validity
                    depth_shape = self.reproj_depths_param.shape
                    if from_idx >= depth_shape[0]:
                        continue
                    
                    if y_coords.max() >= depth_shape[1] or x_coords.max() >= depth_shape[2]:
                        continue
                    
                    # Calculate geometric flow
                    depth_from = self.reproj_depths_param[from_idx, y_coords, x_coords]
                    
                    # Check depth validity
                    if torch.any(depth_from <= 0) or torch.any(torch.isnan(depth_from)) or torch.any(torch.isinf(depth_from)):
                        valid_depth_mask = (depth_from > 0) & torch.isfinite(depth_from)
                        if not valid_depth_mask.any():
                            continue
                        # Filter valid depths
                        y_coords = y_coords[valid_depth_mask]
                        x_coords = x_coords[valid_depth_mask]
                        depth_from = depth_from[valid_depth_mask]
                    R_from, t_from = self.get_poses(from_idx)
                    R_to, t_to = self.get_poses(to_idx)
                    K = self.get_intrinsics_K([from_idx])[0]
                    
                    # Ensure tensor dimensions are correct
                    if R_from.dim() > 2:
                        R_from = R_from.squeeze()
                    if R_to.dim() > 2:
                        R_to = R_to.squeeze()
                    if t_from.dim() > 1:
                        t_from = t_from.squeeze()
                    if t_to.dim() > 1:
                        t_to = t_to.squeeze()
                    
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    
                    x_norm = (x_coords.float() - cx) / fx
                    y_norm = (y_coords.float() - cy) / fy
                    
                    points_3d_from = torch.stack([
                        x_norm * depth_from,
                        y_norm * depth_from,
                        depth_from
                    ], dim=1)  # [N, 3]
                    

                    
                    # Convert to target frame
                    # Step 1: From source camera coordinate system to world coordinate system
                    # P_world = R_from^T * P_cam_from + t_from
                    points_world = points_3d_from @ R_from.T + t_from.unsqueeze(0)  # [N, 3]
                    
                    # Step 2: From world coordinate system to target camera coordinate system
                    # P_cam_to = R_to * (P_world - t_to)
                    points_3d_to = (points_world - t_to.unsqueeze(0)) @ R_to  # [N, 3]
                    
                    # Get predicted flow - Check boundaries first
                    # flow_pred format is [H, W, 2]
                    H, W = flow_pred.shape[0], flow_pred.shape[1]
                    
                    # Check if coordinates are within valid range
                    valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
                    if not valid_mask.all():
                        # Filter valid coordinates
                        y_coords = y_coords[valid_mask]
                        x_coords = x_coords[valid_mask] 
                        depth_from = depth_from[valid_mask]
                        points_3d_from = points_3d_from[valid_mask]
                        points_3d_to = points_3d_to[valid_mask]
                        
                        if len(y_coords) == 0:
                            continue
                    
                    # Recalculate geometric flow (based on filtered coordinates)
                    x_proj = points_3d_to[:, 0] / points_3d_to[:, 2] * fx + cx
                    y_proj = points_3d_to[:, 1] / points_3d_to[:, 2] * fy + cy
                    
                    # Calculate geometric flow (pixel level)
                    geometric_flow_u = x_proj - x_coords.float()  # pixel unit
                    geometric_flow_v = y_proj - y_coords.float()  # pixel unit
                    
                    # Get predicted flow (coordinates are now valid)
                    pred_flow_u = flow_pred[y_coords, x_coords, 0]  # x-direction flow (pixel level)
                    pred_flow_v = flow_pred[y_coords, x_coords, 1]  # y-direction flow (pixel level)
                    
                    # Output geometric flow and predicted flow statistics
                    if epoch == 0:  # Output logs only in first epoch to avoid spam
                        geometric_flow_magnitude = torch.sqrt(geometric_flow_u**2 + geometric_flow_v**2)
                        pred_flow_magnitude = torch.sqrt(pred_flow_u**2 + pred_flow_v**2)
                        
                        print(f"\nFrame pair {from_idx}->{to_idx} flow stats:")
                        print(f"  Geometric flow - Range: [{geometric_flow_magnitude.min():.3f}, {geometric_flow_magnitude.max():.3f}], Mean: {geometric_flow_magnitude.mean():.3f}")
                        print(f"  Predicted flow - Range: [{pred_flow_magnitude.min():.3f}, {pred_flow_magnitude.max():.3f}], Mean: {pred_flow_magnitude.mean():.3f}")
                        print(f"  Geometric flow U - Range: [{geometric_flow_u.min():.3f}, {geometric_flow_u.max():.3f}], Mean: {geometric_flow_u.mean():.3f}")
                        print(f"  Geometric flow V - Range: [{geometric_flow_v.min():.3f}, {geometric_flow_v.max():.3f}], Mean: {geometric_flow_v.mean():.3f}")
                        print(f"  Predicted flow U - Range: [{pred_flow_u.min():.3f}, {pred_flow_u.max():.3f}], Mean: {pred_flow_u.mean():.3f}")
                        print(f"  Predicted flow V - Range: [{pred_flow_v.min():.3f}, {pred_flow_v.max():.3f}], Mean: {pred_flow_v.mean():.3f}")
                        
                        # Add more debug info
                        print(f"  Depth value range: [{depth_from.min():.3f}, {depth_from.max():.3f}], Mean: {depth_from.mean():.3f}")
                        print(f"  Proj coord x_proj - Range: [{x_proj.min():.3f}, {x_proj.max():.3f}], Mean: {x_proj.mean():.3f}")
                        print(f"  Proj coord y_proj - Range: [{y_proj.min():.3f}, {y_proj.max():.3f}], Mean: {y_proj.mean():.3f}")
                        print(f"  Orig coord x_coords - Range: [{x_coords.float().min():.3f}, {x_coords.float().max():.3f}], Mean: {x_coords.float().mean():.3f}")
                        print(f"  Orig coord y_coords - Range: [{y_coords.float().min():.3f}, {y_coords.float().max():.3f}], Mean: {y_coords.float().mean():.3f}")
                    
                    # Check if reverse flow is available
                    reverse_key = f"{to_idx}_{from_idx}"
                    use_cycle_consistency = reverse_key in flow_predictions
                    

                    
                    if use_cycle_consistency:
                        # Use true reverse flow to calculate cycle consistency
                        reverse_flow = flow_predictions[reverse_key]
                        
                        # flow_pred format is [H, W, 2]
                        flow_u = flow_pred[y_coords, x_coords, 0]  # x-direction flow
                        flow_v = flow_pred[y_coords, x_coords, 1]  # y-direction flow
                        
                        target_x = x_coords.float() + flow_u
                        target_y = y_coords.float() + flow_v
                        
                        H, W = flow_pred.shape[0], flow_pred.shape[1]
                        valid_mask = (target_x >= 0) & (target_x < W-1) & (target_y >= 0) & (target_y < H-1)
                        
                        target_x_clamp = torch.clamp(target_x, 0, W-1)
                        target_y_clamp = torch.clamp(target_y, 0, H-1)
                        target_x_int = target_x_clamp.long()
                        target_y_int = target_y_clamp.long()
                        
                        # Use true reverse flow - reverse_flow format is also [H, W, 2]
                        back_flow_u = reverse_flow[target_y_int, target_x_int, 0]  # x-direction reverse flow
                        back_flow_v = reverse_flow[target_y_int, target_x_int, 1]  # y-direction reverse flow
                        
                        cycle_error_x = torch.abs(target_x + back_flow_u - x_coords.float())
                        cycle_error_y = torch.abs(target_y + back_flow_v - y_coords.float())
                        cycle_error = torch.sqrt(cycle_error_x ** 2 + cycle_error_y ** 2)
                        
                        # Weight calculation
                        sigma = getattr(self.opt, 'cycle_consistency_sigma', 1.0)
                        min_weight = getattr(self.opt, 'min_cycle_weight', 0.01)
                        
                        weights = torch.exp(-cycle_error ** 2 / (2 * sigma ** 2))
                        weights = torch.clamp(weights, min=min_weight)
                        weights[~valid_mask] = min_weight
                    else:
                        # Use uniform weights
                        weights = torch.ones(len(y_coords), device=self.device)
                    
                    # Calculate flow loss
                    flow_diff_u = geometric_flow_u - pred_flow_u
                    flow_diff_v = geometric_flow_v - pred_flow_v
                    flow_error = torch.sqrt(flow_diff_u ** 2 + flow_diff_v ** 2)
                    

                    
                    # Use Huber loss
                    if getattr(self.opt, 'use_huber_loss', True):
                        huber_delta = getattr(self.opt, 'huber_delta', 1.0)
                        loss = torch.where(flow_error < huber_delta,
                                         0.5 * flow_error ** 2,
                                         huber_delta * (flow_error - 0.5 * huber_delta))
                    else:
                        loss = flow_error ** 2
                    
                    # Apply weights
                    weighted_loss = loss * weights
                    pair_loss = weighted_loss.mean()
                    
                    if not torch.isnan(pair_loss) and not torch.isinf(pair_loss):
                        total_loss = total_loss + pair_loss
                        valid_pairs += 1
                        
                except Exception as e:
                    continue
            
            if valid_pairs > 0:
                total_loss = total_loss / valid_pairs * flow_weight
                
                if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    total_loss.backward()
                    
                    if "reproj_depth" in self.optimizers:
                        torch.nn.utils.clip_grad_norm_(self.reproj_depths_param, 1.0)
                        self.optimizers["reproj_depth"].step()
                    
                    with torch.no_grad():
                        self.reproj_depths_param.data.clamp_(min=0.01, max=100.0)
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: loss={total_loss.item():.6f}, valid_pairs={valid_pairs}")
        print("Simplified flow optimization completed")