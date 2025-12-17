#!/usr/bin/env python3
"""
Batch Pipeline with QVQ API: Convert all image folders to videos, then execute QVQ API stage1 and stage2 (stage2 can be skipped)
"""

import os
import sys
import glob
import json
import argparse
import subprocess
import time
import logging
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

class BatchQVQPipeline:

    def __init__(self, config):
        self.config = config
        self.results_summary = []
        self.base_frame_dir = config.get('base_frame_dir', '/giga_eval_plat_nas/users/chenxin.li/projects/SAM3R/sa2va/video')

    def setup_folder_logger(self, folder_output_dir, folder_name):
        """Set up logger for individual folder"""
        log_file = os.path.join(folder_output_dir, f"processing_log_{folder_name}.log")
        logger = logging.getLogger(f"folder_{folder_name}")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]: logger.removeHandler(handler)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def find_image_folders(self, root_dir):
        """Find all scene folders in root directory, these folders should contain a subfolder named 'rgb'"""
        image_folders = []
        print(f"Searching for scenes with 'rgb' subfolder in: {root_dir}")
        if not os.path.isdir(root_dir):
            print(f"Error: Root directory not found: {root_dir}")
            return []
        for scene_name in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_name)
            if os.path.isdir(scene_path):
                rgb_folder = os.path.join(scene_path, 'rgb')
                if os.path.isdir(rgb_folder):
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
                    try:
                        has_images = any(f.lower().endswith(tuple(image_extensions)) for f in os.listdir(rgb_folder))
                        if has_images:
                            image_folders.append(rgb_folder)
                            print(f"Found image folder for scene '{scene_name}': {rgb_folder}")
                    except OSError as e:
                         print(f"Warning: Cannot read directory {rgb_folder}: {e}")
        print(f"Total found {len(image_folders)} image folders")
        return sorted(image_folders)

    def check_stage2_completed(self, segmentation_dir, logger=None):
        """Check if Stage 2 has been completed"""
        if not os.path.exists(segmentation_dir): return False
        instance_labels_file = os.path.join(segmentation_dir, "instance_labels.json")
        if not os.path.exists(instance_labels_file): return False
        segmentation_seg_dir = os.path.join(segmentation_dir, "frames", "segmented")
        segmentation_masks_dir = os.path.join(segmentation_dir, "frames", "masks")
        # More robust check, handle cases where directories don't exist
        segmented_files = []
        mask_files = []
        try:
            if os.path.isdir(segmentation_seg_dir):
                segmented_files = [f for f in os.listdir(segmentation_seg_dir) if f.endswith('.jpg')]
            if os.path.isdir(segmentation_masks_dir):
                 mask_files = [f for f in os.listdir(segmentation_masks_dir) if f.endswith('.png')]
        except OSError as e:
            if logger: logger.warning(f"Unable to read stage2 directory contents: {e}")
            return False # If unable to read, consider not completed

        has_results = len(segmented_files) > 0 and len(mask_files) > 0
        if logger:
            logger.info(f"Stage 2 completion status check:")
            logger.info(f"  Segmentation directory: {segmentation_dir}")
            logger.info(f"  instance_labels.json: {'Exists' if os.path.exists(instance_labels_file) else 'Does not exist'}")
            logger.info(f"  Number of segmented images: {len(segmented_files)}")
            logger.info(f"  Number of mask files: {len(mask_files)}")
            logger.info(f"  Stage 2 status: {'Completed' if has_results else 'Not completed'}")
        return has_results

    def create_video_from_images(self, image_folder, output_video_path, fps=20, logger=None):
        """Convert image folder to video"""
        if logger: logger.info(f"Starting to create video from image folder: {image_folder}")
        print(f"\n=== Creating video from {image_folder} ===")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(image_folder, ext)))
            images.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        images = sorted(images)
        if not images:
            error_msg = f"No image files found in {image_folder}"
            if logger: logger.error(error_msg)
            print(error_msg)
            return False
        if logger: logger.info(f"Found {len(images)} images")
        print(f"Found {len(images)} images")
        try:
            frame = cv2.imread(images[0])
            if frame is None:
                error_msg = f"Cannot read first image: {images[0]}"
                if logger: logger.error(error_msg)
                print(error_msg)
                return False
            height, width, layers = frame.shape
            if logger: logger.info(f"Video resolution: {width}x{height}")
            print(f"Video resolution: {width}x{height}")
        except Exception as e:
            error_msg = f"Error reading first image: {e}"
            if logger: logger.error(error_msg)
            print(error_msg)
            return False
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            for i, image_path in enumerate(images):
                frame = cv2.imread(image_path)
                if frame is not None:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video_writer.write(frame)
                    if (i + 1) % 10 == 0:
                        progress_msg = f"Processed {i + 1}/{len(images)} images"
                        if logger: logger.info(progress_msg)
                        print(f"  Processed {i + 1}/{len(images)} images")
                else:
                    warning_msg = f"Warning: Cannot read image {image_path}"
                    if logger: logger.warning(warning_msg)
                    print(f"  {warning_msg}")
            video_writer.release()
            success_msg = f"Video saved to: {output_video_path}"
            if logger: logger.info(success_msg)
            print(f"Video saved to: {output_video_path}")
            return True
        except Exception as e:
            error_msg = f"Error creating video: {e}"
            if logger: logger.error(error_msg)
            print(error_msg)
            return False

    def run_stage1_qvq(self, image_folder, output_json_path, logger=None):
        """Execute Stage 1: QVQ API dynamic object analysis"""
        if logger:
            logger.info(f"Starting to execute Stage 1: QVQ API dynamic object analysis")
            logger.info(f"Input image folder: {image_folder}")
            logger.info(f"Output JSON: {output_json_path}")
        print(f"\n=== Running Stage 1: QVQ API Analysis ===")
        print(f"Frames: {image_folder}")
        print(f"Output JSON: {output_json_path}")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            error_msg = "DASHSCOPE_API_KEY environment variable not set"
            if logger: logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        if logger: logger.info(f"API Key check: {'✅ Set' if api_key else '❌ Not set'} (length: {len(api_key) if api_key else 0})")
        print(f"API Key: {'✅ Set' if api_key else '❌ Not set'} (length: {len(api_key) if api_key else 0})")
        try:
            cmd = [
                sys.executable, "stage1_qwen.py",
                "--frames_path", image_folder,
                "--output_json", output_json_path,
                "--key_frame_dir", self.config.get('key_frame_dir', image_folder),
                "--max_frames", str(25)
            ]
            if logger: logger.info(f"Executing command: {' '.join(cmd)}")
            print(f"Command: {' '.join(cmd)}")
            env = os.environ.copy()
            env["DASHSCOPE_API_KEY"] = api_key
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
            if result.returncode == 0:
                success_msg = "QVQ API Stage 1 completed successfully"
                if logger:
                    logger.info(success_msg)
                    logger.info(f"Stage 1 output:\n{result.stdout}")
                print(f"✅ {success_msg}")
                return True
            else:
                error_msg = f"QVQ API Stage 1 failed with return code: {result.returncode}"
                if logger:
                    logger.error(error_msg)
                    logger.error(f"Stage 1 stdout:\n{result.stdout}")
                    logger.error(f"Stage 1 stderr:\n{result.stderr}")
                print(f"❌ QVQ API Stage 1 failed:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            error_msg = "QVQ API Stage 1 timed out after 60 minutes"
            if logger: logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"QVQ API Stage 1 error: {e}"
            if logger: logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False

    def run_stage2(self, image_folder, video_path, json_path, output_dir, num_frames=None, logger=None):
        """Execute Stage 2: Sa2VA segmentation (using image folder mode)"""
        if logger:
            logger.info(f"Starting to execute Stage 2: Sa2VA segmentation")
            logger.info(f"Image folder: {image_folder}")
            logger.info(f"Video path: {video_path}")
            logger.info(f"JSON path: {json_path}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Frame limit: {'all frames' if num_frames is None else num_frames}")
        print(f"\n=== Running Stage 2: Sa2VA Segmentation (Images Mode) ===")
        print(f"Image Folder: {image_folder}")
        print(f"Video: {video_path}")
        print(f"JSON: {json_path}")
        print(f"Output: {output_dir}")
        print(f"Frame limit: {'All frames' if num_frames is None else num_frames}")
        try:
            if logger: logger.info(f"Using image folder: {image_folder}")
            print(f"Using image folder: {image_folder}")
            cmd = [
                sys.executable, "stage2_sa2va.py",
                "--images",
                "--image_folder", image_folder,
                "--video_fps", str(self.config.get('video_fps', 10)),
                "dummy_input",
                json_path,
                output_dir,
            ]
            if num_frames is not None:
                cmd.extend(["--num_frames", str(num_frames)])
            if logger: logger.info(f"Executing command: {' '.join(cmd)}")
            print(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                success_msg = "Stage 2 completed successfully"
                if logger:
                    logger.info(success_msg)
                    logger.info(f"Stage 2 output:\n{result.stdout}")
                print(f"✅ {success_msg}")
                return True
            else:
                error_msg = f"Stage 2 failed with return code: {result.returncode}"
                if logger:
                    logger.error(error_msg)
                    logger.error(f"Stage 2 stdout:\n{result.stdout}")
                    logger.error(f"Stage 2 stderr:\n{result.stderr}")
                print(f"❌ Stage 2 failed:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            error_msg = "Stage 2 timed out after 60 minutes"
            if logger: logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Stage 2 error: {e}"
            if logger: logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False

    def process_single_folder(self, image_folder, base_output_dir):
        """Process single image folder"""
        scene_path = os.path.dirname(image_folder)
        folder_name = os.path.basename(scene_path)
        print(f"\n{'='*60}")
        print(f"Processing scene: {folder_name}")
        print(f"{'='*60}")
        folder_output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        logger = self.setup_folder_logger(folder_output_dir, folder_name)
        videos_dir = os.path.join(folder_output_dir, "videos")
        analysis_dir = os.path.join(folder_output_dir, "analysis")
        segmentation_dir = os.path.join(folder_output_dir, "segmentation")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(segmentation_dir, exist_ok=True)
        video_path = os.path.join(videos_dir, f"{folder_name}.mp4")
        json_path = os.path.join(analysis_dir, f"dynamic_objects_{folder_name}.json")
        json_exists = os.path.exists(json_path) and os.path.getsize(json_path) > 0
        stage2_completed = self.check_stage2_completed(segmentation_dir, logger)
        
        # ### MODIFIED: Check if Stage 2 should be skipped ###
        skip_stage2_flag = self.config.get('skip_stage2', False)

        result = {
            "folder_name": folder_name, "image_folder": image_folder,
            "video_path": video_path, "json_path": json_path,
            "segmentation_dir": segmentation_dir, "status": "failed",
            "stages_completed": [], "error_message": None, "processing_time": 0
        }
        start_time = time.time()
        logger.info(f"Starting to process folder: {folder_name}")
        logger.info(f"Input image folder: {image_folder}")
        logger.info(f"Output directory: {folder_output_dir}")
        logger.info(f"JSON file status: {'Exists' if json_exists else 'Does not exist, will generate'}")
        logger.info(f"Stage 2 status: {'Completed' if stage2_completed else 'Not completed'}")
        # ### MODIFIED: Record skip_stage2 status ###
        logger.info(f"Skip Stage 2 (--skip_stage2): {'Yes' if skip_stage2_flag else 'No'}")
        logger.info(f"Configuration: {self.config}")
        
        # ### MODIFIED: Modified skip logic ###
        # If JSON exists and (Stage 2 is completed or set to skip Stage 2), skip entire scene
        if json_exists and (stage2_completed or skip_stage2_flag):
            skip_msg = f"Scene has been processed or Stage 2 is being skipped, skipping entire scene: {folder_name}"
            print(f"⏭️  {skip_msg}")
            logger.info(skip_msg)
            result["status"] = "success"
            if json_exists: result["stages_completed"].append("stage1_skipped")
            if stage2_completed: result["stages_completed"].append("stage2_skipped")
            if skip_stage2_flag and not stage2_completed: result["stages_completed"].append("stage2_skipped_by_flag")
            result["processing_time"] = time.time() - start_time
            logger.info(f"Skip processing completed! Time taken: {result['processing_time']:.1f} seconds")
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            return result

        try:
            # Step 1: Create video & Step 2: Run Stage 1 (if JSON doesn't exist)
            if not json_exists:
                logger.info("Step 1: Starting to create video...")
                # if not self.create_video_from_images(image_folder, video_path, self.config['fps'], logger):
                #     result["error_message"] = "Failed to create video"
                #     logger.error(f"Step 1 failed: {result['error_message']}")
                #     return result # Return early
                # result["stages_completed"].append("video_creation")
                # logger.info("Step 1 completed: Video creation successful")
                
                logger.info("Step 2: Starting to run Stage 1 (QVQ API analysis)...")
                if not self.run_stage1_qvq(image_folder, json_path, logger):
                    result["error_message"] = "QVQ API Stage 1 failed"
                    logger.error(f"Step 2 failed: {result['error_message']}")
                    return result # Return early
                result["stages_completed"].append("stage1_qvq")
                logger.info("Step 2 completed: Stage 1 QVQ API analysis successful")
            else:
                skip_msg = f"JSON file already exists, skipping Stage 1: {json_path}"
                print(f"⏭️  {skip_msg}")
                logger.info(skip_msg)
                result["stages_completed"].append("stage1_skipped")
                
                # Even if skipping Stage 1, ensure video exists (if Stage 2 needs to run)
                if not skip_stage2_flag and not stage2_completed and not os.path.exists(video_path):
                    logger.info("Video file does not exist, creating video for Stage 2...")
                    if not self.create_video_from_images(image_folder, video_path, self.config['fps'], logger):
                        result["error_message"] = "Failed to create video for Stage 2"
                        logger.error(f"Video creation failed: {result['error_message']}")
                        return result # Return early
                    result["stages_completed"].append("video_creation")
                    logger.info("Video creation completed")

            # Step 3: Run Stage 2 (check if completed or skipped by parameter)
            # ### MODIFIED: Complete Stage 2 skip/execution logic ###
            if skip_stage2_flag:
                skip_msg = f"Stage 2 is being skipped by --skip_stage2 parameter"
                print(f"⏭️  {skip_msg}")
                logger.info(skip_msg)
                result["stages_completed"].append("stage2_skipped_by_flag")
            elif not stage2_completed: # Only execute if flag is not set and stage2 is not completed
                logger.info("Step 3: Starting to run Stage 2 (Sa2VA segmentation)...")
                if not self.run_stage2(image_folder, video_path, json_path, segmentation_dir, self.config['num_frames'], logger):
                    result["error_message"] = "Stage 2 failed"
                    logger.error(f"Step 3 failed: {result['error_message']}")
                    return result # Return early
                result["stages_completed"].append("stage2")
                logger.info("Step 3 completed: Stage 2 segmentation successful")
            else: # flag not set and stage2 already completed
                skip_msg = f"Stage 2 already completed, skipping segmentation: {segmentation_dir}"
                print(f"⏭️  {skip_msg}")
                logger.info(skip_msg)
                result["stages_completed"].append("stage2_skipped")
            
            result["status"] = "success"
            logger.info(f"Processing completed successfully! Total time: {time.time() - start_time:.1f} seconds")
            
        except Exception as e:
            result["error_message"] = f"Unexpected error: {e}"
            logger.error(f"Exception occurred during processing: {result['error_message']}")
        
        finally:
            result["processing_time"] = time.time() - start_time
            logger.info(f"Processing completed:")
            logger.info(f"  Status: {result['status']}")
            logger.info(f"  Completed stages: {result['stages_completed']}")
            logger.info(f"  Total processing time: {result['processing_time']:.1f} seconds")
            if result["error_message"]: logger.info(f"  Error message: {result['error_message']}")
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        return result

    def run_batch_processing(self, root_image_dir, output_base_dir):
        """Execute batch processing"""
        print(f"🚀 Starting QVQ API batch processing pipeline")
        print(f"📁 Input directory: {root_image_dir}")
        print(f"📁 Output directory: {output_base_dir}")
        print(f"⚙️ Configuration: {self.config}")
        print()
        os.makedirs(output_base_dir, exist_ok=True)
        image_folders = self.find_image_folders(root_image_dir)
        if not image_folders:
            print("❌ No image folders found!")
            return
        total_start_time = time.time()
        for i, image_folder in enumerate(image_folders):
            print(f"\n📋 Processing {i+1}/{len(image_folders)}: {os.path.basename(os.path.dirname(image_folder))}") # Print scene name
            result = self.process_single_folder(image_folder, output_base_dir)
            self.results_summary.append(result)
            if result["status"] == "success":
                stages_info = ""
                # ### MODIFIED: Update description of success status ###
                if "stage1_skipped" in result["stages_completed"] and \
                   ("stage2_skipped" in result["stages_completed"] or "stage2_skipped_by_flag" in result["stages_completed"]):
                   stages_info = " (Fully completed or skipped)"
                elif "stage1_skipped" in result["stages_completed"]:
                   stages_info = " (Stage 1 skipped)"
                elif "stage2_skipped" in result["stages_completed"]:
                   stages_info = " (Stage 2 skipped - previously completed)"
                elif "stage2_skipped_by_flag" in result["stages_completed"]:
                   stages_info = " (Stage 2 skipped by flag)"

                print(f"✅ Completed in {result['processing_time']:.1f}s{stages_info}")
            else:
                print(f"❌ Failed: {result['error_message']}")
        self.generate_summary_report(output_base_dir, time.time() - total_start_time)

    def generate_summary_report(self, output_base_dir, total_time):
        """Generate processing result summary report"""
        print(f"\n{'='*80}")
        print(f"🎯 QVQ API BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        successful = [r for r in self.results_summary if r["status"] == "success"]
        failed = [r for r in self.results_summary if r["status"] == "failed"]
        stage1_processed = [r for r in successful if "stage1_qvq" in r["stages_completed"]]
        stage1_skipped = [r for r in successful if "stage1_skipped" in r["stages_completed"]]
        stage2_processed = [r for r in successful if "stage2" in r["stages_completed"]]
        stage2_skipped = [r for r in successful if "stage2_skipped" in r["stages_completed"]]
        # ### MODIFIED: Add flag skip counting ###
        stage2_skipped_by_flag = [r for r in successful if "stage2_skipped_by_flag" in r["stages_completed"]]
        fully_skipped = [r for r in successful if "stage1_skipped" in r["stages_completed"] and ("stage2_skipped" in r["stages_completed"] or "stage2_skipped_by_flag" in r["stages_completed"])]
        
        print(f"📊 Total folders processed: {len(self.results_summary)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"   - Fully completed or skipped scenes: {len(fully_skipped)}")
        print(f"   - Stage 1 processed: {len(stage1_processed)}")
        print(f"   - Stage 1 skipped (JSON exists): {len(stage1_skipped)}")
        print(f"   - Stage 2 processed: {len(stage2_processed)}")
        print(f"   - Stage 2 skipped (previously completed): {len(stage2_skipped)}")
        # ### MODIFIED: Display flag skip information ###
        print(f"   - Stage 2 skipped (by --skip_stage2 flag): {len(stage2_skipped_by_flag)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"⏱️ Total processing time: {total_time:.1f}s")
        print()

        # ... (detailed list printing can remain unchanged, or also add flag skip information) ...
        # (For brevity, detailed list modification is omitted here, but you can add flag skip information similarly)

        output_base_dir_path = Path(output_base_dir)
        tail_dir_name = output_base_dir_path.name
        report_path = output_base_dir_path.parent / f"{tail_dir_name}_qvq_processing_report.json"
        report_data = {
            "summary": {
                "total_folders": len(self.results_summary), "successful": len(successful),
                "fully_skipped": len(fully_skipped), "stage1_processed": len(stage1_processed),
                "stage1_skipped": len(stage1_skipped), "stage2_processed": len(stage2_processed),
                "stage2_skipped": len(stage2_skipped), 
                "stage2_skipped_by_flag": len(stage2_skipped_by_flag), # ### MODIFIED: Add to report ###
                "failed": len(failed), "total_time": total_time,
                "timestamp": datetime.now().isoformat(), "stage1_method": "QVQ API"
            },
            "configuration": self.config, "detailed_results": self.results_summary
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="QVQ API Batch Pipeline: Process multiple image folders through video creation, QVQ API stage1, and stage2 (using frame folder mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (use default directories, process all frames)
  python batch_process_qwen_pipeline.py

  # Specify custom directories
  python batch_process_qwen_pipeline.py /path/to/image/folders /path/to/output

  # Limit frames and skip Stage 2
  python batch_process_qwen_pipeline.py /path/to/image/folders /path/to/output --num_frames 30 --skip_stage2

  # Custom parameters and frame folder directory
  python batch_process_qwen_pipeline.py /path/to/image/folders /path/to/output --fps 30 --num_frames 64 --base_frame_dir /custom/frame/dir --video_fps 15

Note:
  - Make sure to set DASHSCOPE_API_KEY environment variable before running
  - QVQ API has rate limits, processing may be slower than local models
        """
    )
    parser.add_argument("input_dir", nargs="?", default="/data/workspace_hyz/projects/dynamicBA/data/ego_demo1", help="Root directory containing image folders")
    parser.add_argument("output_dir", nargs="?", default="/data/workspace_hyz/projects/dynamicBA/data/ego_demo1", help="Output directory for all results")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS for input video creation (default: 30)")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames for stage2 (default: None - process all frames)")
    parser.add_argument("--base_frame_dir", default="/data/workspace_hyz/projects/dynamicBA/data/ego_demo1", help="Base directory containing frame folders")
    parser.add_argument("--key_frame_dir", default="/data/workspace_hyz/projects/dynamicBA/data/ego_demo1", help="Key_frame directory")
    parser.add_argument("--video_fps", type=int, default=30, help="Output segmentation video frame rate (default: 30)")
    
    # ### MODIFIED: Add --skip_stage2 parameter ###
    parser.add_argument("--skip_stage2",
                        action="store_true", # Set as flag
                        help="Skip Stage 2 (Sa2VA segmentation) processing.")

    args = parser.parse_args()
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ Error: DASHSCOPE_API_KEY environment variable not set!")
        print("Please set your API key: export DASHSCOPE_API_KEY=your_api_key")
        sys.exit(1)
    
    print(f"📋 QVQ API Configuration:")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Base frame directory: {args.base_frame_dir}")
    print(f"   Key frame directory: {args.key_frame_dir}")
    print(f"   Input video FPS: {args.fps}")
    print(f"   Output video FPS: {args.video_fps}")
    print(f"   Number of frames: {'All frames' if args.num_frames is None else args.num_frames}")
    # ### MODIFIED: Print skip_stage2 status ###
    print(f"   Skip Stage 2: {'Yes' if args.skip_stage2 else 'No'}")
    print(f"   API Key: {'✅ Set' if os.getenv('DASHSCOPE_API_KEY') else '❌ Not set'}")
    print()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not os.path.exists(args.base_frame_dir):
        print(f"❌ Base frame directory not found: {args.base_frame_dir}")
        print(f"Please ensure the frame folders exist in the specified directory")
        sys.exit(1)
    
    # ### MODIFIED: Add skip_stage2 to config ###
    config = {
        "fps": args.fps,
        "num_frames": args.num_frames,
        "base_frame_dir": args.base_frame_dir,
        "key_frame_dir": args.key_frame_dir,
        "video_fps": args.video_fps,
        "skip_stage2": args.skip_stage2 # Add parameter value to config
    }
    
    pipeline = BatchQVQPipeline(config)
    pipeline.run_batch_processing(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()