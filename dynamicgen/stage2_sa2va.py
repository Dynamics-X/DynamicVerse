#!/usr/bin/env python3
"""
Stage 2: Sa2VA Segmentation
Based on Stage 1 generated dynamic object JSON, use Sa2VA for segmentation and visualization
Supports direct reading from specified directory frame folders, no need to extract frames from video
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import decord
from decord import VideoReader, cpu
import argparse
import logging
from datetime import datetime

# Sa2VA imports  
from transformers import AutoModelForCausalLM, AutoTokenizer as Sa2VATokenizer

# Set GPU for Sa2VA stage (single GPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Sa2VASegmenter:
    """Stage 2: Use Sa2VA for segmentation"""
    def __init__(self, sa2va_model_path, output_dir=None, gpu_id=None):
        print("=== Stage 2: Initializing Sa2VA Segmenter ===")

        # Automatically select GPU
        if gpu_id is None:
            # Read GPU ID from environment variable
            # gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
            gpu_id = 0

        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Set up logger
        self.logger = self.setup_logger(output_dir)
        self.logger.info(f"Starting to initialize Sa2VA segmenter (device: {self.device})")

        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info(f"Cleaned GPU {gpu_id} memory")

            self.logger.info(f"GPU {gpu_id} memory cleaned")
        self.logger.info("Starting to load Sa2VA model...")
        print(f"Loading Sa2VA model on {self.device}...")
        self.sa2va_model = AutoModelForCausalLM.from_pretrained(
            sa2va_model_path,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.sa2va_tokenizer = Sa2VATokenizer.from_pretrained(
            sa2va_model_path,
            trust_remote_code=True
        )
        self.logger.info(f"Sa2VA model loaded successfully (device: {self.device})")
        print(f"Sa2VA model loaded successfully on {self.device}")
        print("=========================================\n")

    def setup_logger(self, output_dir=None):
        """Set up logger"""
        logger = logging.getLogger("Sa2VASegmenter")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if output directory specified)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, f"stage2_segmentation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Log file created: {log_file}")
        
        return logger

    def find_frame_folder(self, base_frame_dir, video_name_or_path):
        """Find corresponding frame folder from base directory"""
        print(f"=== Finding frame folder for: {video_name_or_path} ===")
        
        # If input is video path, extract filename (without extension)
        if video_name_or_path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_name = os.path.splitext(os.path.basename(video_name_or_path))[0]
        else:
            video_name = video_name_or_path
        
        print(f"Looking for frame folder matching: {video_name}")
        
        # Possible folder name patterns
        possible_names = [
            video_name,
            f"{video_name}_frames",
            f"frames_{video_name}",
            video_name.lower(),
            f"{video_name.lower()}_frames",
            f"frames_{video_name.lower()}"
        ]
        
        # Look for matching folder in base directory
        for name in possible_names:
            candidate_path = os.path.join(base_frame_dir, name)
            if os.path.exists(candidate_path) and os.path.isdir(candidate_path):
                # Check if folder contains image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
                has_images = False
                for file in os.listdir(candidate_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        has_images = True
                        break
                
                if has_images:
                    print(f"Found frame folder: {candidate_path}")
                    return candidate_path
        
        print(f"No matching frame folder found in {base_frame_dir}")
        print(f"Available folders: {os.listdir(base_frame_dir) if os.path.exists(base_frame_dir) else 'Directory not found'}")
        return None

    def extract_frames_from_video(self, video_path, num_frames=32, temp_dir="temp_frames"):
        """Extract frames from video"""
        print(f"=== Extracting frames from video: {video_path} ===")
        
        # Create temporary directory
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Read video using decord
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            print(f"Total video frames: {total_frames}")
            
            # Uniformly sample frames
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            
            # Convert to PIL Image and save
            vid_frames = []
            frame_paths = []
            
            for i, frame in enumerate(frames):
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame)
                vid_frames.append(pil_frame)
                
                # Save to temp directory (optional, for debugging)
                frame_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
                pil_frame.save(frame_path)
                frame_paths.append(frame_path)
            
            print(f"Extracted {len(vid_frames)} frames successfully")
            return vid_frames, frame_paths
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return None, None

    def segment_and_visualize_from_frames(self, base_frame_dir, video_name, dynamic_objects_json_path, output_dir, num_frames=None, video_fps=10):
        """Perform segmentation and visualization from specified frame folder"""
        print("=== Starting Sa2VA Segmentation from Frame Folder ===")
        
        # Find corresponding frame folder
        frame_folder = self.find_frame_folder(base_frame_dir, video_name)
        if frame_folder is None:
            print("Failed to find frame folder. Exiting.")
            return False
        
        # Load dynamic object analysis results
        with open(dynamic_objects_json_path, 'r', encoding='utf-8') as f:
            dynamic_objects_result = json.load(f)
        
        if not dynamic_objects_result['dynamic']:
            print("No dynamic objects found in JSON. Exiting.")
            return False
        
        print(f"Found {len(dynamic_objects_result['dynamic'])} dynamic objects to segment:")
        for obj in dynamic_objects_result['dynamic']:
            print(f"  - {obj}")
        
        # Load images from frame folder
        vid_frames, frame_paths = self.load_images_from_folder(frame_folder, num_frames)
        if not vid_frames:
            print("Failed to load frames. Exiting.")
            return False
            
        # Get image size
        image_shape = np.array(vid_frames[0]).shape
        
        # Segment each dynamic object
        all_masks = []
        object_names = []
        
        for obj_name in dynamic_objects_result['dynamic']:
            print(f"\n--- Processing object: {obj_name} ---")
            # Get corresponding reasoning description
            reasoning_description = dynamic_objects_result['reasoning'].get(obj_name, "")
            masks = self.segment_object(vid_frames, obj_name, reasoning_description)
            all_masks.append(masks)
            object_names.append(obj_name)
        
        # Filter out failed segmentations
        valid_masks = []
        valid_names = []
        for masks, name in zip(all_masks, object_names):
            if masks is not None:
                valid_masks.append(masks)
                valid_names.append(name)
        
        if not valid_masks:
            print("No valid segmentation results. Exiting.")
            return False
        
        # Create instance segmentation
        instance_masks, instance_labels = self.create_instance_segmentation(
            valid_masks, valid_names, image_shape
        )
        
        if instance_masks is None:
            print("Failed to create instance segmentation. Exiting.")
            return False
        
        # Visualize results
        self.visualize_results(vid_frames, instance_masks, instance_labels, output_dir, video_fps)
        
        print("=== Sa2VA Segmentation Complete ===")
        return True

    def segment_and_visualize_from_video(self, video_path, dynamic_objects_json_path, output_dir, num_frames=32, video_fps=10):
        """Directly segment and visualize from video"""
        print("=== Starting Sa2VA Segmentation from Video ===")
        
        # Load dynamic object analysis results
        with open(dynamic_objects_json_path, 'r', encoding='utf-8') as f:
            dynamic_objects_result = json.load(f)
        
        if not dynamic_objects_result['dynamic']:
            print("No dynamic objects found in JSON. Exiting.")
            return False
        
        print(f"Found {len(dynamic_objects_result['dynamic'])} dynamic objects to segment:")
        for obj in dynamic_objects_result['dynamic']:
            print(f"  - {obj}")
        
        # Extract frames from video
        vid_frames, frame_paths = self.extract_frames_from_video(video_path, num_frames)
        if vid_frames is None:
            print("Failed to extract frames. Exiting.")
            return False
            
        # Get image size
        image_shape = np.array(vid_frames[0]).shape
        
        # Segment each dynamic object
        all_masks = []
        object_names = []
        
        for obj_name in dynamic_objects_result['dynamic']:
            print(f"\n--- Processing object: {obj_name} ---")
            # Get corresponding reasoning description
            reasoning_description = dynamic_objects_result['reasoning'].get(obj_name, "")
            masks = self.segment_object(vid_frames, obj_name, reasoning_description)
            all_masks.append(masks)
            object_names.append(obj_name)
        
        # Filter out failed segmentations
        valid_masks = []
        valid_names = []
        for masks, name in zip(all_masks, object_names):
            if masks is not None:
                valid_masks.append(masks)
                valid_names.append(name)
        
        if not valid_masks:
            print("No valid segmentation results. Exiting.")
            return False
        
        # Create instance segmentation
        instance_masks, instance_labels = self.create_instance_segmentation(
            valid_masks, valid_names, image_shape
        )
        
        if instance_masks is None:
            print("Failed to create instance segmentation. Exiting.")
            return False
        
        # Visualize results
        self.visualize_results(vid_frames, instance_masks, instance_labels, output_dir, video_fps)
        
        # Clean up temporary files
        self.cleanup_temp_files("temp_frames")
        
        print("=== Sa2VA Segmentation Complete ===")
        return True

    def cleanup_temp_files(self, temp_dir):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")

    def segment_and_visualize(self, image_folder, dynamic_objects_json_path, output_dir, video_fps=10, num_frames=None):
        """Segment and visualize based on dynamic object JSON (keep original method for compatibility)"""
        self.logger.info("Starting Sa2VA segmentation processing")
        self.logger.info(f"Input image folder: {image_folder}")
        self.logger.info(f"Dynamic object JSON: {dynamic_objects_json_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Video FPS: {video_fps}")
        print("=== Starting Sa2VA Segmentation ===")
        
        # Load dynamic object analysis results
        with open(dynamic_objects_json_path, 'r', encoding='utf-8') as f:
            dynamic_objects_result = json.load(f)
        
        if not dynamic_objects_result['dynamic']:
            self.logger.warning("No dynamic objects found in JSON, exiting processing")
            print("No dynamic objects found in JSON. Exiting.")
            return False
        
        self.logger.info(f"Found {len(dynamic_objects_result['dynamic'])} dynamic objects to segment:")
        print(f"Found {len(dynamic_objects_result['dynamic'])} dynamic objects to segment:")
        for obj in dynamic_objects_result['dynamic']:
            self.logger.info(f"  - {obj}")
            print(f"  - {obj}")
        
        # Load images
        vid_frames, _ = self.load_images_from_folder(image_folder, max_frames=num_frames)
        if not vid_frames:
            self.logger.error("No images found, exiting processing")
            print("No images found. Exiting.")
            return False
            
        # Get image size
        image_shape = np.array(vid_frames[0]).shape
        self.logger.info(f"Image size: {image_shape}")
        
        # Segment each dynamic object
        all_masks = []
        object_names = []
        
        for obj_name in dynamic_objects_result['dynamic']:
            self.logger.info(f"Starting processing object: {obj_name}")
            print(f"\n--- Processing object: {obj_name} ---")
            # Get corresponding reasoning description
            reasoning_description = dynamic_objects_result['reasoning'].get(obj_name, "")
            masks = self.segment_object(vid_frames, obj_name, reasoning_description)
            all_masks.append(masks)
            object_names.append(obj_name)
        
        # Filter out failed segmentations
        valid_masks = []
        valid_names = []
        for masks, name in zip(all_masks, object_names):
            if masks is not None:
                valid_masks.append(masks)
                valid_names.append(name)
                self.logger.info(f"Object {name} segmentation successful")
            else:
                self.logger.warning(f"Object {name} segmentation failed")
        
        if not valid_masks:
            self.logger.error("No valid segmentation results, exiting processing")
            print("No valid segmentation results. Exiting.")
            return False
        
        self.logger.info(f"Valid segmentation results: {len(valid_masks)} objects")
        
        # Create instance segmentation
        instance_masks, instance_labels = self.create_instance_segmentation(
            valid_masks, valid_names, image_shape
        )
        
        if instance_masks is None:
            self.logger.error("Failed to create instance segmentation, exiting processing")
            print("Failed to create instance segmentation. Exiting.")
            return False
        
        # Visualize results
        self.visualize_results(vid_frames, instance_masks, instance_labels, output_dir, video_fps)
        
        self.logger.info("Sa2VA segmentation processing completed")
        print("=== Sa2VA Segmentation Complete ===")
        return True

    def load_images_from_folder(self, image_folder, max_frames=None):
        """Load images from folder, supports frame limit"""
        self.logger.info(f"Starting to load images from folder: {image_folder}")
        self.logger.info(f"Max frame limit: {max_frames if max_frames else 'None'}")
        
        image_files = []
        image_paths = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
        
        for filename in sorted(list(os.listdir(image_folder))):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(filename)
                image_paths.append(os.path.join(image_folder, filename))

        self.logger.info(f"Found {len(image_paths)} images in {image_folder}")
        print(f"Found {len(image_paths)} images in {image_folder}")
        
        # If max frames specified, perform uniform sampling
        if max_frames is not None and len(image_paths) > max_frames:
            self.logger.info(f"Sampling {max_frames} frames from {len(image_paths)} images")
            print(f"Sampling {max_frames} frames from {len(image_paths)} total frames")
            indices = np.linspace(0, len(image_paths) - 1, num=max_frames, dtype=int)
            image_paths = [image_paths[i] for i in indices]
            image_files = [image_files[i] for i in indices]
        elif max_frames is None:
            self.logger.info(f"Processing all {len(image_paths)} frames")
            print(f"Processing all {len(image_paths)} frames")
        else:
            self.logger.info(f"Processing all {len(image_paths)} frames (less than max_frames={max_frames})")
            print(f"Processing all {len(image_paths)} frames (less than max_frames={max_frames})")

        vid_frames = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            vid_frames.append(img)
            
        self.logger.info(f"Successfully loaded {len(vid_frames)} images")
        print(f"Loaded {len(vid_frames)} images from {image_folder}")
        return vid_frames, image_paths

    def segment_object(self, vid_frames, object_name, reasoning_description=""):
        """Segment specified object using Sa2VA"""
        self.logger.info(f"Starting to segment object: {object_name}")
        print(f"Segmenting object: {object_name}")
        
        # Construct segmentation prompt - use detailed reasoning description as guide
        if reasoning_description:
            text = f"<image>Please help me segment the {object_name}. Additional details: {reasoning_description}"
            # text = f"<image>Please help me segment the {object_name}. "
            self.logger.info(f"Using detailed description: {reasoning_description[:100]}...")
            print(f"  Using detailed description: {reasoning_description[:100]}...")
        else:
            text = f"<image>Please help me segment the {object_name}."
            self.logger.info("Using basic object name only")
            print(f"  Using basic object name only")
        

        self.logger.info(f"Calling Sa2VA model for segmentation...")
        result = self.sa2va_model.predict_forward(
            video=vid_frames,
            text=text,
            tokenizer=self.sa2va_tokenizer,
        )
        
        prediction = result['prediction']
        self.logger.info(f"Model prediction result: {prediction}")
        print(f"  Prediction: {prediction}")
        
        if '[SEG]' in prediction and 'prediction_masks' in result:
            masks = result['prediction_masks'][0]  # First segmentation result
            self.logger.info(f"Generated {len(masks)} masks")
            print(f"  Generated {len(masks)} masks")
            return masks
        else:
            self.logger.warning(f"No segmentation mask generated for object {object_name}")
            print(f"  No segmentation mask generated for {object_name}")
            return None
                


    def create_instance_segmentation(self, all_masks, object_names, image_shape):
        """Integrate multiple binary masks into instance segmentation results"""
        print("=== Creating Instance Segmentation ===")
        
        # Get image size
        if len(all_masks) == 0:
            print("No masks to process")
            return None, {}
            
        num_frames = len(all_masks[0])
        instance_masks = []
        instance_labels = {}
        
        for frame_idx in range(num_frames):
            # Create instance segmentation image (H, W), 0 for background, 1, 2, 3... for different instances
            instance_mask = np.zeros(image_shape[:2], dtype=np.int32)
            
            instance_id = 1
            for mask_idx, (masks, obj_name) in enumerate(zip(all_masks, object_names)):
                if masks is not None and frame_idx < len(masks):
                    binary_mask = masks[frame_idx]
                    
                    # Convert binary mask to current instance ID
                    if isinstance(binary_mask, torch.Tensor):
                        binary_mask = binary_mask.cpu().numpy()
                    
                    # Ensure mask is boolean type
                    if binary_mask.dtype != bool:
                        binary_mask = binary_mask > 0.5
                    
                    # Set current object area to instance ID
                    instance_mask[binary_mask] = instance_id
                    instance_labels[instance_id] = obj_name
                    instance_id += 1
            
            instance_masks.append(instance_mask)
        
        print(f"Created instance segmentation with {len(instance_labels)} instances")
        for inst_id, label in instance_labels.items():
            print(f"  Instance {inst_id}: {label}")
        print("===================================\n")
        
        return instance_masks, instance_labels

    def visualize_results(self, vid_frames, instance_masks, instance_labels, output_dir, video_fps=10):
        """Visualize instance segmentation results and generate video"""
        self.logger.info("Starting to visualize segmentation results")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Video FPS: {video_fps}")
        self.logger.info(f"Total frames: {len(vid_frames)}")
        self.logger.info(f"Instance count: {len(instance_labels)}")
        print("=== Visualizing Results ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sub-folder structure
        frames_dir = os.path.join(output_dir, "frames")
        videos_dir = os.path.join(output_dir, "videos")
        
        original_frames_dir = os.path.join(frames_dir, "original")
        mask_frames_dir = os.path.join(frames_dir, "masks")
        segmented_frames_dir = os.path.join(frames_dir, "segmented")
        overlay_frames_dir = os.path.join(frames_dir, "overlay")
        
        for dir_path in [frames_dir, videos_dir, original_frames_dir, mask_frames_dir, segmented_frames_dir, overlay_frames_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info("Output folder structure creation completed")
        
        # Generate color map
        num_instances = len(instance_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, max(num_instances, 12)))
        self.logger.info(f"Generating color map, instance count: {num_instances}")
        
        # Collect all frame data for video generation
        original_video_frames = []
        mask_video_frames = []
        segmented_video_frames = []
        overlay_video_frames = []
        
        self.logger.info("Starting processing each frame...")
        for frame_idx, (frame, instance_mask) in enumerate(zip(vid_frames, instance_masks)):
            # Convert PIL image to numpy array
            frame_np = np.array(frame)
            
            # 1. Save original frame
            original_path = os.path.join(original_frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(original_path, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            original_video_frames.append(frame_np)
            
            # 2. Create colored mask
            colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
            for instance_id, label in instance_labels.items():
                mask_area = (instance_mask == instance_id)
                if np.any(mask_area):
                    color = (colors[instance_id-1][:3] * 255).astype(np.uint8)
                    colored_mask[mask_area] = color
            
            # 3. Save pure mask frame
            # mask_path = os.path.join(mask_frames_dir, f"frame_{frame_idx:05d}.png")
            # cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            # mask_video_frames.append(colored_mask)
            # 3. Save pure mask frame
            mask_path = os.path.join(mask_frames_dir, f"frame_{frame_idx:05d}.png")
            try:
                # === Add check ===
                image_to_save = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(mask_path, image_to_save)
                if not success:
                    self.logger.warning(f"Warning: Unable to save mask frame to {mask_path}")
                    print(f"⚠️ Warning: Unable to save mask frame to {mask_path}") # Print to console as well
                # === Check end ===
                else:
                    mask_video_frames.append(colored_mask) # Add to video frame list only after successful save

            except Exception as e:
                # Catch possible underlying error (although imwrite usually doesn't throw)
                self.logger.error(f"Error: Exception occurred while saving mask frame to {mask_path}: {e}")
                print(f"❌ Error: Exception occurred while saving mask frame to {mask_path}: {e}")
                
            # 4. Create segmentation overlay (semi-transparent)
            alpha = 0.6
            overlay = frame_np.copy()
            mask_area = instance_mask > 0
            overlay[mask_area] = (alpha * colored_mask[mask_area] + (1-alpha) * frame_np[mask_area]).astype(np.uint8)
            
            # Save overlay frame
            overlay_path = os.path.join(overlay_frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_video_frames.append(overlay)
            
            # 5. Create segmented image (show segmented area only, black background)
            segmented = np.zeros_like(frame_np)
            for instance_id, label in instance_labels.items():
                mask_area = (instance_mask == instance_id)
                if np.any(mask_area):
                    segmented[mask_area] = frame_np[mask_area]
            
            # Save segmented frame
            segmented_path = os.path.join(segmented_frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(segmented_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
            segmented_video_frames.append(segmented)
            
            # Log progress every 100 frames
            if (frame_idx + 1) % 100 == 0:
                self.logger.info(f"Processed {frame_idx + 1}/{len(vid_frames)} frames")
        
        self.logger.info("All frames processed, starting video generation...")
        
        # Generate video
        self._create_video_from_frames(original_video_frames, os.path.join(videos_dir, "original.mp4"), video_fps)
        self._create_video_from_frames(mask_video_frames, os.path.join(videos_dir, "masks.mp4"), video_fps)
        self._create_video_from_frames(overlay_video_frames, os.path.join(videos_dir, "overlay.mp4"), video_fps)
        self._create_video_from_frames(segmented_video_frames, os.path.join(videos_dir, "segmented.mp4"), video_fps)
        
        # Save label mapping
        labels_path = os.path.join(output_dir, "instance_labels.json")
        with open(labels_path, 'w') as f:
            json.dump(instance_labels, f, indent=2)
        
        self.logger.info(f"Saving instance label mapping: {labels_path}")
        
        # Generate result summary
        summary = {
            "total_frames": len(vid_frames),
            "total_instances": len(instance_labels),
            "instance_labels": instance_labels,
            "video_fps": video_fps,
            "output_structure": {
                "frames": {
                    "original": original_frames_dir,
                    "masks": mask_frames_dir,
                    "segmented": segmented_frames_dir,
                    "overlay": overlay_frames_dir
                },
                "videos": {
                    "original": os.path.join(videos_dir, "original.mp4"),
                    "masks": os.path.join(videos_dir, "masks.mp4"),
                    "segmented": os.path.join(videos_dir, "segmented.mp4"),
                    "overlay": os.path.join(videos_dir, "overlay.mp4")
                }
            }
        }
        
        summary_path = os.path.join(output_dir, "result_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saving result summary: {summary_path}")
        self.logger.info(f"Visualization result generation completed:")
        self.logger.info(f"  📁 Original frames: {original_frames_dir}")
        self.logger.info(f"  📁 Mask frames: {mask_frames_dir}")
        self.logger.info(f"  📁 Segmented frames: {segmented_frames_dir}")
        self.logger.info(f"  📁 Overlay frames: {overlay_frames_dir}")
        self.logger.info(f"  🎬 Original video: {os.path.join(videos_dir, 'original.mp4')}")
        self.logger.info(f"  🎬 Mask video: {os.path.join(videos_dir, 'masks.mp4')}")
        self.logger.info(f"  🎬 Segmented video: {os.path.join(videos_dir, 'segmented.mp4')}")
        self.logger.info(f"  🎬 Overlay video: {os.path.join(videos_dir, 'overlay.mp4')}")
        
        print(f"✅ Saved {len(vid_frames)} frames to different folders:")
        print(f"   📁 Original frames: {original_frames_dir}")
        print(f"   📁 Mask frames: {mask_frames_dir}")
        print(f"   📁 Segmented frames: {segmented_frames_dir}")
        print(f"   📁 Overlay frames: {overlay_frames_dir}")
        print(f"✅ Generated videos (FPS: {video_fps}):")
        print(f"   🎬 Original video: {os.path.join(videos_dir, 'original.mp4')}")
        print(f"   🎬 Masks video: {os.path.join(videos_dir, 'masks.mp4')}")
        print(f"   🎬 Segmented video: {os.path.join(videos_dir, 'segmented.mp4')}")
        print(f"   🎬 Overlay video: {os.path.join(videos_dir, 'overlay.mp4')}")
        print(f"📄 Instance labels: {labels_path}")
        print(f"📄 Result summary: {summary_path}")
        print("===========================\n")

    def _create_video_from_frames(self, frames, output_path, fps=10):
        """Create video from frame list"""
        if not frames:
            warning_msg = f"Warning: No frames to create video {output_path}"
            self.logger.warning(warning_msg)
            print(warning_msg)
            return False
        
        try:
            # Get frame size
            height, width = frames[0].shape[:2]
            self.logger.info(f"Creating video: {output_path}, Size: {width}x{height}, FPS: {fps}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write all frames
            for frame in frames:
                # Ensure correct frame format
                if len(frame.shape) == 3:  # RGB
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:  # Grayscale
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                video_writer.write(frame_bgr)
            
            video_writer.release()
            self.logger.info(f"Video creation successful: {output_path}")
            print(f"✅ Created video: {output_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error creating video {output_path}: {e}"
            self.logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Sa2VA Segmentation - Multiple input modes for video segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Frame folder mode (using existing frame folders):
  python stage2_sa2va_segmentation.py --frames --base_frame_dir /path/to/video/frames --video_name bus.mp4 /path/to/dynamic_objects.json /path/to/output

  # Video input mode:
  python stage2_sa2va_segmentation.py /path/to/video.mp4 /path/to/dynamic_objects.json /path/to/output 32
  
  # Image folder mode:
  python stage2_sa2va_segmentation.py --images --image_folder /path/to/images /path/to/dynamic_objects.json /path/to/output
        """
    )
    
    # Switches for different input modes
    parser.add_argument("--images", action="store_true", help="Use image folder input mode")
    parser.add_argument("--frames", action="store_true", help="Use existing frame folder mode")
    
    # Image folder mode parameters
    parser.add_argument("--image_folder", default="/giga_eval_plat_nas/users/chenxin.li/projects/SAM3R/sa2va/video/bus", help="Path to image folder (required when using --images)")
    
    # Frame folder mode parameters
    parser.add_argument("--base_frame_dir", help="Base directory containing frame folders (required when using --frames)")
    parser.add_argument("--video_name", help="Video name or identifier to find corresponding frame folder (required when using --frames)")
    
    # Video generation parameters
    parser.add_argument("--video_fps", type=int, default=10, help="Output video frame rate (default: 10)")
    
    # Positional arguments - for video mode
    parser.add_argument("input_path", nargs="?", default="/giga_eval_plat_nas/users/chenxin.li/projects/SAM3R/sa2va/video/bus.mp4", help="Path to input video file (video mode) or will be ignored (other modes)")
    parser.add_argument("dynamic_objects_json", default="/giga_eval_plat_nas/users/chenxin.li/projects/SAM3R/sa2va/integrated_output/dynamic_objects_analysis_bus.json", nargs="?", help="Path to dynamic objects JSON file from stage 1")
    parser.add_argument("output_dir", nargs="?", default="/giga_eval_plat_nas/users/chenxin.li/projects/SAM3R/sa2va/segmentation/segmentation_output_bus", help="Output directory for segmentation results") 
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to process (optional, default: None, process all frames)")
    
    args = parser.parse_args()
    
    # Configure Sa2VA model path
    sa2va_model_path = "../preprocess/pretrained/Sa2Va-8B"
    sa2va_model_path = "/mnt/shared-storage-user/wenkairun/idc2-shared/wenkairun/ckpts/Sa2Va-8B"
    
    # Check for mutually exclusive mode options
    modes_count = sum([args.images, args.frames])
    if modes_count > 1:
        print("Error: Only one input mode can be specified (--images, --frames, or default video mode)")
        sys.exit(1)
    
    # Initialize Sa2VA segmenter
    segmenter = Sa2VASegmenter(sa2va_model_path, args.output_dir)
    success = False
    
    if args.frames:
        # Frame folder mode
        if not args.base_frame_dir or not args.video_name:
            print("Error: --base_frame_dir and --video_name are required when using --frames mode")
            sys.exit(1)
            
        if not args.dynamic_objects_json or not args.output_dir:
            print("Error: dynamic_objects_json and output_dir are required")
            parser.print_help()
            sys.exit(1)
            
        if not os.path.exists(args.base_frame_dir):
            print(f"Error: Base frame directory not found: {args.base_frame_dir}")
            sys.exit(1)
            
        if not os.path.exists(args.dynamic_objects_json):
            print(f"Error: Dynamic objects JSON file not found: {args.dynamic_objects_json}")
            print("Please run stage1 first!")
            sys.exit(1)
        
        # Skip scene with existing segmentation results
        result_json = os.path.join(args.output_dir, "result_summary.json")
        if os.path.exists(result_json):
            print(f"[Skip] Segmentation already exists for this scene: {args.output_dir}")
            return
        
        print(f"Running in frame folder mode:")
        print(f"  Base frame dir: {args.base_frame_dir}")
        print(f"  Video name: {args.video_name}")
        print(f"  JSON file: {args.dynamic_objects_json}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Max frames: {args.num_frames}")
        print(f"  Video FPS: {args.video_fps}")
        
        success = segmenter.segment_and_visualize_from_frames(
            args.base_frame_dir, args.video_name, args.dynamic_objects_json, args.output_dir, args.num_frames, args.video_fps
        )
        
    elif args.images:
        # Image folder mode
        if not args.image_folder:
            print("Error: --image_folder is required when using --images mode")
            sys.exit(1)
            
        if not args.dynamic_objects_json or not args.output_dir:
            print("Error: dynamic_objects_json and output_dir are required")
            parser.print_help()
            sys.exit(1)
            
        if not os.path.exists(args.image_folder):
            print(f"Error: Image folder not found: {args.image_folder}")
            sys.exit(1)
            
        if not os.path.exists(args.dynamic_objects_json):
            print(f"Error: Dynamic objects JSON file not found: {args.dynamic_objects_json}")
            print("Please run stage1 first!")
            sys.exit(1)
        
        # # Skip scene with existing segmentation results
        # result_json = os.path.join(args.output_dir, "result_summary.json")
        # if os.path.exists(result_json):
        #     print(f"[Skip] Segmentation already exists for this scene: {args.output_dir}")
        #     return
        
        print(f"Running in image folder mode:")
        print(f"  Image folder: {args.image_folder}")
        print(f"  JSON file: {args.dynamic_objects_json}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Video FPS: {args.video_fps}")
        
        success = segmenter.segment_and_visualize(args.image_folder, args.dynamic_objects_json, args.output_dir, args.video_fps, args.num_frames)
        
    else:
        # Default video input mode
        if not args.input_path or not args.dynamic_objects_json or not args.output_dir:
            print("Error: input_path, dynamic_objects_json, and output_dir are required for video mode")
            parser.print_help()
            sys.exit(1)
            
        if not os.path.exists(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            sys.exit(1)
            
        if not os.path.exists(args.dynamic_objects_json):
            print(f"Error: Dynamic objects JSON file not found: {args.dynamic_objects_json}")
            print("Please run stage1 first!")
            sys.exit(1)
        
        # Skip scene with existing segmentation results
        result_json = os.path.join(args.output_dir, "result_summary.json")
        if os.path.exists(result_json):
            print(f"[Skip] Segmentation already exists for this scene: {args.output_dir}")
            return
        
        print(f"Running in video mode:")
        print(f"  Video file: {args.input_path}")
        print(f"  JSON file: {args.dynamic_objects_json}")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Frames to extract: {args.num_frames}")
        print(f"  Video FPS: {args.video_fps}")
        
        success = segmenter.segment_and_visualize_from_video(
            args.input_path, args.dynamic_objects_json, args.output_dir, args.num_frames, args.video_fps
        )
    
    if success:
        print(f"Stage 2 complete! Results saved to: {args.output_dir}")
    else:
        print("Stage 2 failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()