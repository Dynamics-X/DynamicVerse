#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample frames with large motion from video based on optical flow changes

This script evaluates inter-frame motion intensity based on optical flow calculations and selects frames with the largest motion changes for output.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import logging
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OpticalFlowFrameSampler:
    """Optical flow based video frame sampler"""
    
    def __init__(self, flow_model_type='unimatch', device='cuda'):
        """
        Initialize optical flow frame sampler
        
        Args:
            flow_model_type (str): Optical flow model type, supports 'unimatch' or 'flow_anything'
            device (str): Computing device
        """
        self.flow_model_type = flow_model_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.flow_processor = None
        self.frames = []  # resized frames for optical flow calculation
        self.original_frames = []  # original size frames for final saving
        self.flow_magnitudes = []
        
        logging.info(f"Initialize optical flow frame sampler, using device: {self.device}")
    
    def load_flow_model(self, flow_anything_ckpt_path=None, flow_anything_config_path=None):
        """
        Load pretrained optical flow model
        
        Args:
            flow_anything_ckpt_path (str): Flow-Anything model checkpoint path
            flow_anything_config_path (str): Flow-Anything configuration file path
        """
        logging.info(f"Loading optical flow model: {self.flow_model_type}")
        
        if self.flow_model_type == 'opencv':
            # Use OpenCV optical flow algorithm, no extra model loading needed
            self.flow_processor = 'opencv'
            logging.info("Using OpenCV optical flow algorithm")
        
        elif self.flow_model_type == 'unimatch':
            try:
                from unimatch.unimatch_flow import FlowOcclusionProcessor
                model_path = 'unimatch/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth'
                self.flow_processor = FlowOcclusionProcessor(flow_model=self.flow_model_type, model_path=model_path)
                self.flow_processor = self.flow_processor.to(self.device)
                logging.info("Successfully loaded Unimatch optical flow processor")
            except ImportError as e:
                logging.error(f"Failed to import Unimatch module: {e}")
                logging.warning("Automatically switching to OpenCV optical flow algorithm")
                self.flow_model_type = 'opencv'
                self.flow_processor = 'opencv'
                logging.info("Switched to OpenCV optical flow algorithm")
            except Exception as e:
                logging.error(f"Failed to load Unimatch optical flow model: {e}")
                logging.warning("Automatically switching to OpenCV optical flow algorithm")
                self.flow_model_type = 'opencv'
                self.flow_processor = 'opencv'
                logging.info("Switched to OpenCV optical flow algorithm")
        
        elif self.flow_model_type == 'flow_anything':
            try:
                from flow_anything_wrapper import get_flow_anything_model
                
                if not flow_anything_ckpt_path or not flow_anything_config_path:
                    raise ValueError("Using flow_anything model requires providing ckpt_path and config_path")
                
                self.flow_processor = get_flow_anything_model(
                    ckpt_path=flow_anything_ckpt_path,
                    config_path=flow_anything_config_path,
                    device=self.device
                )
                self.flow_processor = self.flow_processor.to(self.device)
                logging.info("Successfully loaded Flow-Anything model")
            except ImportError as e:
                logging.error(f"Failed to import Flow-Anything module: {e}")
                logging.warning("Automatically switching to OpenCV optical flow algorithm")
                self.flow_model_type = 'opencv'
                self.flow_processor = 'opencv'
                logging.info("Switched to OpenCV optical flow algorithm")
            except Exception as e:
                logging.error(f"Failed to load Flow-Anything model: {e}")
                logging.warning("Automatically switching to OpenCV optical flow algorithm")
                self.flow_model_type = 'opencv'
                self.flow_processor = 'opencv'
                logging.info("Switched to OpenCV optical flow algorithm")
        
        else:
            raise ValueError(f"Unsupported optical flow model type: {self.flow_model_type}")
    
    def load_video_frames(self, video_path: str, max_frames: int = None, 
                         target_size: Tuple[int, int] = None) -> List[np.ndarray]:
        """
        Load frames from video file
        
        Args:
            video_path (str): Video file path
            max_frames (int): Maximum frames to load
            target_size (Tuple[int, int]): Target size (width, height)
            
        Returns:
            List[np.ndarray]: List of frames
        """
        logging.info(f"Processing loading video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video information: total frames={total_frames}, FPS={fps}")
        
        # Clear previous frame data
        frames = []
        self.original_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save original size frame
            self.original_frames.append(original_frame.copy())
            
            # Resize if needed for optical flow calculation
            if target_size is not None:
                resized_frame = cv2.resize(original_frame, target_size)
                frames.append(resized_frame)
            else:
                frames.append(original_frame)
            
            frame_idx += 1
            
            if max_frames and frame_idx >= max_frames:
                break
            
            if frame_idx % 100 == 0:
                logging.info(f"Loaded {frame_idx} frames")
        
        cap.release()
        logging.info(f"Loading completed, total {len(frames)} frames")
        logging.info(f"original frame count: {len(self.original_frames)}, processing frame count: {len(frames)}")
        self.frames = frames
        return frames
    
    def get_flow_prediction(self, img_from: np.ndarray, img_to: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow between two frames
        
        Args:
            img_from (np.ndarray): source frame (H, W, 3)
            img_to (np.ndarray): target frame (H, W, 3)
            
        Returns:
            np.ndarray: optical flow (H, W, 2)
        """
        if self.flow_processor is None:
            raise RuntimeError("Optical flow model not loaded yet, please call load_flow_model() first")
        
        if self.flow_model_type == 'unimatch':
            # Input format expected by Unimatch
            img_from_tensor = torch.from_numpy(img_from).float().to(self.device) / 255.0
            img_to_tensor = torch.from_numpy(img_to).float().to(self.device) / 255.0
            img_from_tensor = img_from_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            img_to_tensor = img_to_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
            img_pair = torch.cat([img_from_tensor, img_to_tensor], dim=1)
            
            with torch.no_grad():
                results = self.flow_processor(img_pair)
            
            flow = results[0, 0, 3:5] if not isinstance(results, tuple) else results[0][0, 0, 3:5]
            flow = flow.permute(1, 2, 0).cpu().numpy()
            
        elif self.flow_model_type == 'flow_anything':
            # Flow-Anything processing
            from flow_anything_wrapper import load_image_flow_anything, calc_flow_anything
            
            image1, _ = load_image_flow_anything(img_from)
            image2, _ = load_image_flow_anything(img_to)
            image1, image2 = image1.to(self.device), image2.to(self.device)
            
            args = getattr(self.flow_processor, 'flow_anything_args', None)
            
            with torch.no_grad():
                flow, _ = calc_flow_anything(args, self.flow_processor, image1, image2)
            
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
        
        elif self.flow_model_type == 'opencv':
            # Use OpenCV Farneback algorithm
            # Convert image to grayscale
            prev_gray = cv2.cvtColor(img_from, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(img_to, cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, 
                None, 
                0.5,  # pyr_scale
                3,    # levels
                15,   # winsize
                3,    # iterations
                5,    # poly_n
                1.2,  # poly_sigma
                0     # flags
            )
        
        return flow
    
    def calculate_flow_magnitude(self, flow: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculate average magnitude of optical flow
        
        Args:
            flow (np.ndarray): optical flow (H, W, 2)
            mask (np.ndarray): optional mask, calculate optical flow only in masked area
            
        Returns:
            float: Average optical flow magnitude
        """
        # Calculate optical flow magnitude sqrt(dx^2 + dy^2)
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        
        if mask is not None:
            magnitude = magnitude[mask > 0]
        
        # Return average magnitude, use robust statistics
        return {
            'mean': float(np.mean(magnitude)),
            'median': float(np.median(magnitude)),
            'percentile_75': float(np.percentile(magnitude, 75)),
            'percentile_90': float(np.percentile(magnitude, 90)),
            'percentile_95': float(np.percentile(magnitude, 95)),
            'max': float(np.max(magnitude))
        }
    
    def compute_frame_motion_scores(self, step: int = 1) -> List[Dict]:
        """
        Calculate motion scores for all adjacent frame pairs
        
        Args:
            step (int): Frame interval, default is 1 (adjacent frames)
            
        Returns:
            List[Dict]: Motion information for each frame pair
        """
        if len(self.frames) < 2:
            raise ValueError("Need at least 2 frames to calculate optical flow")
        
        motion_scores = []
        
        for i in range(0, len(self.frames) - step):
            from_idx = i
            to_idx = i + step
            
            logging.info(f"Calculating optical flow for frame {from_idx} -> {to_idx}")
            
            # Calculate optical flow
            flow = self.get_flow_prediction(self.frames[from_idx], self.frames[to_idx])
            
            # Calculate optical flow magnitude statistics
            magnitude_stats = self.calculate_flow_magnitude(flow)
            
            motion_info = {
                'from_frame': from_idx,
                'to_frame': to_idx,
                'flow_magnitude_stats': magnitude_stats,
                'flow': flow  # Save optical flow for visualization
            }
            
            motion_scores.append(motion_info)
            
            logging.info(f"Frame pair {from_idx}->{to_idx}: Avg optical flow magnitude={magnitude_stats['mean']:.3f}, "
                        f"Median={magnitude_stats['median']:.3f}, "
                        f"95th percentile={magnitude_stats['percentile_95']:.3f}")
        
        self.motion_scores = motion_scores
        return motion_scores
    
    def select_high_motion_frames(self, num_frames: int, 
                                 selection_metric: str = 'percentile_90',
                                 min_frame_gap: int = 5) -> List[int]:
        """
        Select frames with highest motion
        
        Args:
            num_frames (int): Number of frames to select
            selection_metric (str): Selection metric ('mean', 'median', 'percentile_90', 'max', etc.)
            min_frame_gap (int): Minimum interval between selected frames
            
        Returns:
            List[int]: Selected frame indices
        """
        if not hasattr(self, 'motion_scores'):
            raise RuntimeError("Please call compute_frame_motion_scores() first")
        
        # Sort by selection metric
        sorted_scores = sorted(self.motion_scores, 
                             key=lambda x: x['flow_magnitude_stats'][selection_metric], 
                             reverse=True)
        
        selected_frames = []
        
        for score_info in sorted_scores:
            frame_idx = score_info['from_frame']
            
            # Check if there is enough interval with selected frames
            if all(abs(frame_idx - selected) >= min_frame_gap for selected in selected_frames):
                selected_frames.append(frame_idx)
                
                if len(selected_frames) >= num_frames:
                    break
        
        # Sort by frame index
        selected_frames.sort()
        
        logging.info(f"Selected high motion frames: {selected_frames}")
        return selected_frames
    
    def save_selected_frames(self, selected_frame_indices: List[int], 
                           output_dir: str, prefix: str = 'high_motion_frame') -> None:
        """
        Save selected frames to file
        
        Args:
            selected_frame_indices (List[int]): Selected frame indices
            output_dir (str): Output directory
            prefix (str): Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame_idx in enumerate(selected_frame_indices):
            # Check if index is valid
            if frame_idx >= len(self.original_frames):
                logging.error(f"Frame index {frame_idx} out of range, original_frames length is {len(self.original_frames)}")
                continue
            
            # Save using original size frame
            frame = self.original_frames[frame_idx]
            
            # Save as PNG format
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}_frame_{frame_idx:04d}.png")
            
            # Save using PIL
            img_pil = Image.fromarray(frame)
            img_pil.save(output_path)
            
            logging.info(f"Saved original size frame {frame_idx} to {output_path}")
    
    def visualize_motion_analysis(self, output_dir: str, top_k: int = 10) -> None:
        """
        Visualize motion analysis results
        
        Args:
            output_dir (str): Output directory
            top_k (int): Visualize top K frame pairs with highest motion
        """
        if not hasattr(self, 'motion_scores'):
            raise RuntimeError("Please call compute_frame_motion_scores() first")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Plot motion score curve for all frames
        frame_indices = [score['from_frame'] for score in self.motion_scores]
        mean_magnitudes = [score['flow_magnitude_stats']['mean'] for score in self.motion_scores]
        percentile_90_magnitudes = [score['flow_magnitude_stats']['percentile_90'] for score in self.motion_scores]
        
        plt.figure(figsize=(12, 6))
        plt.plot(frame_indices, mean_magnitudes, label='Average Optical Flow Magnitude', marker='o', markersize=3)
        plt.plot(frame_indices, percentile_90_magnitudes, label='90th Percentile Optical Flow Magnitude', marker='s', markersize=3)
        plt.xlabel('Frame Index')
        plt.ylabel('Optical Flow Magnitude')
        plt.title('Inter-frame Motion Intensity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'motion_analysis.png'), dpi=150)
        plt.close()
        
        # 2. Save optical flow visualization for frame pairs with highest motion scores
        sorted_scores = sorted(self.motion_scores, 
                             key=lambda x: x['flow_magnitude_stats']['percentile_90'], 
                             reverse=True)
        
        for i in range(min(top_k, len(sorted_scores))):
            score_info = sorted_scores[i]
            from_frame = score_info['from_frame']
            to_frame = score_info['to_frame']
            flow = score_info['flow']
            
            self._save_flow_visualization(
                self.frames[from_frame], 
                self.frames[to_frame], 
                flow,
                os.path.join(output_dir, f'flow_vis_top_{i+1}_frames_{from_frame}_{to_frame}.png'),
                title=f"Frame {from_frame} -> {to_frame} (Optical Flow Magnitude: {score_info['flow_magnitude_stats']['percentile_90']:.3f})"
            )
    
    def _save_flow_visualization(self, img_from: np.ndarray, img_to: np.ndarray, 
                               flow: np.ndarray, output_path: str, title: str = "") -> None:
        """
        Save optical flow visualization
        
        Args:
            img_from (np.ndarray): source frame
            img_to (np.ndarray): target frame  
            flow (np.ndarray): optical flow
            output_path (str): Output path
            title (str): Image title
        """
        # Calculate optical flow magnitude and angle
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        
        # Create HSV image to visualize optical flow
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi) * 179  # Hue: Direction
        hsv[:, :, 1] = 255  # Saturation: Max
        hsv[:, :, 2] = np.clip(magnitude / magnitude.max() * 255, 0, 255)  # Value: Magnitude
        
        # Convert to RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Create visualization image
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(img_from)
        axes[0, 0].set_title('Source Frame')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_to)
        axes[0, 1].set_title('Target Frame')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(flow_rgb)
        axes[1, 0].set_title('Optical Flow Visualization (HSV)')
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(magnitude, cmap='hot')
        axes[1, 1].set_title('Optical Flow Magnitude')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Optical flow based video frame sampler')
    parser.add_argument('--video_path', type=str, required=True, help='input video path')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--num_frames', type=int, default=10, help='number of frames to sample')
    parser.add_argument('--flow_model', type=str, default='unimatch', 
                       choices=['unimatch', 'flow_anything'], help='optical flow model type')
    parser.add_argument('--max_frames', type=int, default=None, help='maximum processing frame count')
    parser.add_argument('--target_width', type=int, default=None, help='Target width, keep original width if not specified')
    parser.add_argument('--target_height', type=int, default=None, help='Target height, keep original height if not specified')
    parser.add_argument('--min_frame_gap', type=int, default=5, help='minimum interval between selected frames')
    parser.add_argument('--selection_metric', type=str, default='percentile_75',
                       choices=['mean', 'median', 'percentile_75', 'percentile_90', 'percentile_95', 'max'],
                       help='metric for selecting frames')
    parser.add_argument('--frame_step', type=int, default=1, help='frame interval for optical flow calculation')
    
    # Flow-Anything specific parameters
    parser.add_argument('--flow_anything_ckpt', type=str, help='Flow-Anything model checkpoint path')
    parser.add_argument('--flow_anything_config', type=str, help='Flow-Anything configuration file path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize sampler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampler = OpticalFlowFrameSampler(flow_model_type=args.flow_model, device=device)
    
    # Load optical flow model
    if args.flow_model == 'flow_anything':
        sampler.load_flow_model(
            flow_anything_ckpt_path=args.flow_anything_ckpt,
            flow_anything_config_path=args.flow_anything_config
        )
    else:
        sampler.load_flow_model()
    
    # Load video frames
    target_size = (args.target_width, args.target_height) if args.target_width and args.target_height else None
    frames = sampler.load_video_frames(
        args.video_path, 
        max_frames=args.max_frames, 
        target_size=target_size
    )
    
    # Calculate motion scores
    motion_scores = sampler.compute_frame_motion_scores(step=args.frame_step)
    
    # Select high motion frames
    selected_frames = sampler.select_high_motion_frames(
        num_frames=args.num_frames,
        selection_metric=args.selection_metric,
        min_frame_gap=args.min_frame_gap
    )
    
    # Save selected frames
    sampler.save_selected_frames(selected_frames, args.output_dir)
    
    # Generate visualization
    sampler.visualize_motion_analysis(args.output_dir, top_k=min(10, len(motion_scores)))
    
    # Save analysis results
    import json
    analysis_result = {
        'video_path': args.video_path,
        'total_frames': len(frames),
        'selected_frames': selected_frames,
        'selection_metric': args.selection_metric,
        'motion_scores': [
            {
                'from_frame': score['from_frame'],
                'to_frame': score['to_frame'],
                'flow_magnitude_stats': score['flow_magnitude_stats']
            } for score in motion_scores
        ]
    }
    
    with open(os.path.join(args.output_dir, 'analysis_result.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Analysis completed! Result saved in: {args.output_dir}")
    logging.info(f"Selected high motion frames: {selected_frames}")


if __name__ == '__main__':
    main()