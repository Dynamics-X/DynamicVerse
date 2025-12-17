#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset batch processing script
Select keyframes with high optical flow motion from image sequences in each scene
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
import json
import glob
from tqdm import tqdm
import shutil
import time

# Add project root directory to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our optical flow frame sampler
from sample_high_motion_frames import OpticalFlowFrameSampler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MotionFrameExtractor:
    """Processor for extracting high motion keyframes from video frame sequences"""
    
    def __init__(self, input_root: str, output_root: str, flow_model_type: str = 'unimatch'):
        """
        Initialize high motion frame extractor
        
        Args:
            input_root (str): Dataset root directory, containing multiple scene subdirectories
            output_root (str): Output root directory
            flow_model_type (str): Optical flow model type
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.flow_model_type = flow_model_type
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize optical flow sampler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampler = OpticalFlowFrameSampler(flow_model_type=flow_model_type, device=device)
        
        logging.info(f"Initialize frame extractor: {input_root} -> {output_root}")
    
    def find_sequences(self) -> List[Path]:
        """
        Find all folders containing image sequences
        
        Returns:
            List[Path]: List of image sequence folder paths
        """
        sequence_folders = []
        
        # Iterate through all subfolders in the input root (each subfolder is considered a scene)
        for scene_folder in self.input_root.iterdir():
            if scene_folder.is_dir():
                # Assume image frames are located in the "rgb" subdirectory
                # Modify here if your dataset structure is different
                rgb_folder = scene_folder / "rgb"
                if rgb_folder.exists() and rgb_folder.is_dir():
                    sequence_folders.append(rgb_folder)
        
        sequence_folders.sort()  # Sort by name
        logging.info(f"Found {len(sequence_folders)} image sequence folders")
        
        return sequence_folders
    
    def load_rgb_sequence(self, sequence_folder: Path) -> List[np.ndarray]:
        """
        Load image sequence from folder
        
        Args:
            sequence_folder (Path): Folder path containing image sequence
            
        Returns:
            List[np.ndarray]: List of images
        """
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(str(sequence_folder / ext)))
        
        image_files.sort()  # Sort by filename
        
        if not image_files:
            logging.warning(f"No image files found in {sequence_folder}")
            return []
        
        # Loading images
        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        logging.info(f"Loaded {len(images)} images from {sequence_folder.parent.name}")
        return images
    
    def process_single_sequence(self, sequence_folder: Path, 
                              sample_ratio: float = 0.3,
                              selection_metric: str = 'percentile_90',
                              min_frame_gap: int = 2,
                              target_size: Tuple[int, int] = None) -> Dict:
        """
        Process single image sequence
        
        Args:
            sequence_folder (Path): Image sequence folder path
            sample_ratio (float): Sample ratio (e.g., 0.3 means sampling 30% of frames)
            selection_metric (str): Selection metric
            min_frame_gap (int): Minimum frame gap
            target_size (Tuple[int, int]): Target size
            
        Returns:
            Dict: Processing result
        """
        scene_name = sequence_folder.parent.name
        logging.info(f"Start processing scene: {scene_name}")
        start_time = time.time()
        
        try:
            # Loading image sequence
            original_images = self.load_rgb_sequence(sequence_folder)
            if len(original_images) < 2:
                logging.warning(f"Insufficient images in scene {scene_name}, skipping")
                return None
            
            # Set original frames
            self.sampler.original_frames = original_images.copy()
            
            # Resize images (if specified) for optical flow calculation
            if target_size is not None:
                processing_images = []
                for img in original_images:
                    resized_img = cv2.resize(img, target_size)
                    processing_images.append(resized_img)
                self.sampler.frames = processing_images
            else:
                # If resize is not needed, processing frames are original frames
                self.sampler.frames = original_images
            
            # Compute motion scores
            motion_scores = self.sampler.compute_frame_motion_scores(step=1)
            
            # Calculate number of frames to select based on sample ratio
            total_frames = len(original_images)
            num_frames_to_select = max(1, int(total_frames * sample_ratio))
            
            # If too many frames to select, adjust min_frame_gap
            max_possible_frames = total_frames // (min_frame_gap + 1) + 1
            if num_frames_to_select > max_possible_frames:
                # Reduce min_frame_gap to accommodate more frames
                adjusted_min_gap = max(1, (total_frames - 1) // num_frames_to_select)
                logging.info(f"Scene {scene_name}: Adjusting min frame gap from {min_frame_gap} to {adjusted_min_gap} to fit {num_frames_to_select} frames")
                min_frame_gap = adjusted_min_gap
            
            logging.info(f"Scene {scene_name}: Total frames={total_frames}, Sample ratio={sample_ratio:.1%}, Selected frames={num_frames_to_select}")
            
            # Select high motion frames
            selected_frames = self.sampler.select_high_motion_frames(
                num_frames=num_frames_to_select,
                selection_metric=selection_metric,
                min_frame_gap=min_frame_gap
            )
            
            # Create scene output directory and rgb subdirectory
            scene_output_dir = self.output_root / scene_name
            scene_rgb_dir = scene_output_dir / 'rgb'
            scene_rgb_dir.mkdir(parents=True, exist_ok=True)
            
            # Save selected frames to rgb subdirectory
            self.sampler.save_selected_frames(
                selected_frames, 
                str(scene_rgb_dir),
                prefix=f"{scene_name}_high_motion"
            )
            
            # Generate visualization (only save top 3 highest motion frame pairs)
            self.sampler.visualize_motion_analysis(
                str(scene_output_dir), 
                top_k=min(3, len(motion_scores))
            )
            
            # Prepare return result
            actual_sample_ratio = len(selected_frames) / len(original_images)
            result = {
                'scene_name': scene_name,
                'total_frames': len(original_images),
                'selected_frames': selected_frames,
                'target_sample_ratio': sample_ratio,
                'actual_sample_ratio': actual_sample_ratio,
                'selected_frame_count': len(selected_frames),
                'motion_scores_summary': {
                    'total_pairs': len(motion_scores),
                    'avg_motion': np.mean([s['flow_magnitude_stats']['mean'] for s in motion_scores]),
                    'max_motion': np.max([s['flow_magnitude_stats']['percentile_90'] for s in motion_scores])
                }
            }
            
            processing_time = time.time() - start_time
            result['processing_time_seconds'] = processing_time
            logging.info(f"Scene {scene_name} Processing completed, took {processing_time:.2f} seconds. Selected {len(selected_frames)} frames.")
            return result
            
        except Exception as e:
            logging.error(f"Error processing scene {scene_name}: {e}")
            return None
    
    def process_all_sequences(self, 
                            sample_ratio: float = 0.3,
                            selection_metric: str = 'percentile_90',
                            min_frame_gap: int = 2,
                            target_size: Tuple[int, int] = None,
                            max_scenes: int = None) -> List[Dict]:
        """
        Process all image sequences
        
        Args:
            sample_ratio (float): Sample ratio (e.g., 0.3 means sampling 30% of frames)
            selection_metric (str): Selection metric
            min_frame_gap (int): Minimum frame gap
            target_size (Tuple[int, int]): Target size
            max_scenes (int): Maximum number of scenes to process
            
        Returns:
            List[Dict]: All processing results
        """
        # Loading optical flow model
        logging.info("Processing Loading optical flow model...")
        self.sampler.load_flow_model()
        
        # Find all image sequences
        sequence_folders = self.find_sequences()
        
        if max_scenes is not None:
            sequence_folders = sequence_folders[:max_scenes]
            logging.info(f"Limiting number of processed scenes to: {max_scenes}")
        
        # Process each sequence
        all_results = []
        
        for sequence_folder in tqdm(sequence_folders, desc="Processing scenes"):
        
            # --- New Code Start ---
            # Infer scene name and corresponding output path from input path
            scene_name = sequence_folder.parent.name
            scene_output_dir = self.output_root / scene_name
            
            # Check if output directory already exists
            if scene_output_dir.exists():
                logging.info(f"Scene {scene_name} output directory already exists, skipping processing.")
                continue  # Skip current loop, process next scene
            # --- New Code End ---
            
            result = self.process_single_sequence(
                sequence_folder,
                sample_ratio=sample_ratio,
                selection_metric=selection_metric,
                min_frame_gap=min_frame_gap,
                target_size=target_size
            )
            if result is not None:
                all_results.append(result)

        # for sequence_folder in tqdm(sequence_folders, desc="Processing scenes"):
        #     result = self.process_single_sequence(
        #         sequence_folder,
        #         sample_ratio=sample_ratio,
        #         selection_metric=selection_metric,
        #         min_frame_gap=min_frame_gap,
        #         target_size=target_size
        #     )
            
        #     if result is not None:
        #         all_results.append(result)
        
        # Save total results
        self.save_processing_summary(all_results)
        
        return all_results
    
    def save_processing_summary(self, results: List[Dict]) -> None:
        """
        Save processing summary
        
        Args:
            results (List[Dict]): Processing result list
        """
        total_time = sum(r.get('processing_time_seconds', 0) for r in results)
        summary = {
            'dataset_root': str(self.input_root),
            'output_root': str(self.output_root),
            'flow_model_type': self.flow_model_type,
            'processed_scenes': len(results),
            'total_processing_time_seconds': total_time,
            'average_time_per_scene_seconds': total_time / len(results) if results else 0,
            'results': results
        }
        
        # Save JSON file
        summary_file = self.output_root / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generating overall statistics chart
        
        logging.info(f"Processing summary saved to: {summary_file}")
    

    
    def create_merged_dataset(self, copy_original: bool = False) -> None:
        """
        Create merged high motion frame dataset
        
        Args:
            copy_original (bool): Whether to copy original image sequences as well
        """
        merged_dir = self.output_root / "merged_high_motion_frames"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        
        # Collect all high motion frames
        for scene_dir in self.output_root.iterdir():
            if scene_dir.is_dir() and scene_dir.name != "merged_high_motion_frames":
                # Find scene rgb subdirectory
                rgb_dir = scene_dir / 'rgb'
                if rgb_dir.exists():
                    # Find high motion frame files
                    motion_frames = list(rgb_dir.glob("*high_motion*.png"))
                    
                    for frame_file in motion_frames:
                        # Rename and copy to merged directory
                        new_name = f"frame_{frame_count:05d}_{scene_dir.name}_{frame_file.stem}.png"
                        shutil.copy2(frame_file, merged_dir / new_name)
                        frame_count += 1
        
        logging.info(f"Merged dataset creation completed, total {frame_count} high motion frames, saved at: {merged_dir}")


def main():
    """Main function - supports command line arguments or direct configuration run"""
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Use command line argument mode
        parser = argparse.ArgumentParser(description='High motion keyframe extraction tool')
        parser.add_argument('--input_root', type=str, required=True, 
                           help='Dataset root directory, containing multiple scene subdirectories')
        parser.add_argument('--output_root', type=str, required=True,
                           help='Output root directory path')
        parser.add_argument('--flow_model', type=str, default='unimatch',
                           choices=['unimatch', 'flow_anything'], help='Optical flow model type')
        parser.add_argument('--sample_ratio', type=float, default=0.3,
                           help='Sample ratio (e.g., 0.3 means sampling 30% of frames)')
        parser.add_argument('--selection_metric', type=str, default='percentile_90',
                           choices=['mean', 'median', 'percentile_75', 'percentile_90', 'percentile_95', 'max'],
                           help='Metric for selecting frames')
        parser.add_argument('--min_frame_gap', type=int, default=3,
                           help='Minimum gap between selected frames')
        parser.add_argument('--target_width', type=int, default=None,
                           help='Target width, keep original width if not specified')
        parser.add_argument('--target_height', type=int, default=None,
                           help='Target height, keep original height if not specified')
        parser.add_argument('--max_scenes', type=int, default=None,
                           help='Max scenes to process (for testing)')
        parser.add_argument('--create_merged', action='store_true',
                           help='Create merged high motion frame dataset')
        
        args = parser.parse_args()
        
        input_root = args.input_root
        output_root = args.output_root
        flow_model = args.flow_model
        sample_ratio = args.sample_ratio
        selection_metric = args.selection_metric
        min_frame_gap = args.min_frame_gap
        target_size = (args.target_width, args.target_height) if args.target_width and args.target_height else None
        max_scenes = args.max_scenes
        create_merged = args.create_merged
        
    else:
        # Direct configuration mode
        print("=" * 60)
        print("High Motion Frame Sampling Tool")
        print("=" * 60)
        
        # Configuration parameters - You can modify these directly
        input_root = "../data/demo"
        output_root = "../data/demo-extract"
        flow_model = 'unimatch'  # 'unimatch' or 'flow_anything'
        
        # Processing parameters
        sample_ratio = 0.3             # Sample ratio (30% of frames)
        selection_metric = 'percentile_90'  # Selection metric
        min_frame_gap = 2               # Minimum gap between selected frames
        target_size = None              # Target image size, None means keep original size
        max_scenes = None               # Max scenes to process, None means process all scenes
        create_merged = True            # Whether to create merged dataset
        
        print(f"Dataset path: {input_root}")
        print(f"Output path: {output_root}")
        print(f"Optical flow model: {flow_model}")
        print(f"Sample ratio: {sample_ratio:.1%} (sample 30% frames per scene)")
        print(f"Selection metric: {selection_metric}")
        print(f"Minimum frame gap: {min_frame_gap}")
        print(f"Target size: {target_size if target_size else 'Keep original size'}")
        print(f"Max scenes to process: {max_scenes if max_scenes else 'All'}")
        print(f"Create merged dataset: {'Yes' if create_merged else 'No'}")
        print("-" * 60)
        
        # Check input path
        if not os.path.exists(input_root):
            print(f"❌ Error: Dataset path does not exist: {input_root}")
            print("Please modify the input_root variable in the code to the correct path")
            return
        
        print(f"✅ Dataset found: {input_root}")
    
    # Create processor
    print("\n🚀 Starting Processing...")
    processor = MotionFrameExtractor(
        input_root=input_root,
        output_root=output_root,
        flow_model_type=flow_model
    )
    
    # Process all sequences
    start_total_time = time.time()
    results = processor.process_all_sequences(
        sample_ratio=sample_ratio,
        selection_metric=selection_metric,
        min_frame_gap=min_frame_gap,
        target_size=target_size,
        max_scenes=max_scenes
    )
    total_processing_time = time.time() - start_total_time
    
    # Create merged dataset
    if create_merged:
        print("\n📦 Creating merged high motion frame dataset...")
        processor.create_merged_dataset()
    
    # Output result summary
    print("\n" + "=" * 60)
    print("🎉 Processing Completed!")
    print("=" * 60)
    print(f"⏱️  Total time: {total_processing_time:.2f} seconds")
    print(f"✅ Processed {len(results)} scenes in total")
    if results:
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / len(results)
        print(f"⏱️  Average scene processing time: {avg_time:.2f} seconds")
    print(f"📁 Results saved in: {output_root}")
    
    if create_merged:
        print(f"📦 Merged high motion frames saved in: {output_root}/merged_high_motion_frames")
    
    # Display processing result summary
    if results:
        total_original_frames = sum(r['total_frames'] for r in results)
        total_selected_frames = sum(r['selected_frame_count'] for r in results)
        overall_sample_ratio = total_selected_frames / total_original_frames if total_original_frames > 0 else 0
        print(f"🎯 Total selected {total_selected_frames}/{total_original_frames} high motion frames (Overall sample ratio: {overall_sample_ratio:.1%})")
        
        print("\n📊 Processing results per scene:")
        for result in results:
            scene_name = result['scene_name']
            total_frames = result['total_frames']
            selected_count = result['selected_frame_count']
            actual_ratio = result['actual_sample_ratio']
            avg_motion = result['motion_scores_summary']['avg_motion']
            max_motion = result['motion_scores_summary']['max_motion']
            processing_time = result.get('processing_time_seconds', -1)
            print(f"  📹 {scene_name}: {selected_count}/{total_frames} frames ({actual_ratio:.1%}) | Avg Motion: {avg_motion:.3f} | Max Motion: {max_motion:.3f} | Time: {processing_time:.2f}s")
    
    print("\n🔍 For more details, please check:")
    print(f"  - Processing summary: {output_root}/processing_summary.json")
    print(f"  - Statistics chart: {output_root}/overall_statistics.png")



if __name__ == '__main__':
    # Run main program
    main()