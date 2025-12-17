#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
import numpy as np
from PIL import Image
from dam import DescribeAnythingModel, disable_torch_init
import cv2
import glob
import os
import tempfile
import shutil
import csv
import time

def load_instance_masks(image_files, mask_dir, dataset_format='default'):
    """Load PNG format instance segmentation masks from mask_dir."""
    raw_masks = []
    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        
        # Determine mask filename based on dataset format
        if dataset_format == 'ytvis':
            mask_filename = base + '_instance.png'
        elif dataset_format == 'spring':
            # Assume mask filename is identical to frame filename (without extension)
            mask_filename = base + '.png'
        else: # Default case
            mask_filename = base + '.png'
            
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        # Modified: Use .convert('L') to ensure mask image is loaded as single-channel grayscale
        mask_png = np.array(Image.open(mask_path).convert('L')) # <--- Modified code
        raw_masks.append(mask_png)
        
    return raw_masks


def print_streaming(text):
    print(text, end="", flush=True)


def add_contour(img, mask):
    """Add contour to image for visualization."""
    img = img.copy()
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Modified: cv2.findContours returns two values in newer versions
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 'Describe Anything' script with pre-computed multi-instance masks")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Dataset root directory containing all video ID folders.')
    parser.add_argument('--caption_path', type=str, default='./results/captions',
                        help='Directory to save description results')
    parser.add_argument('--dataset_format', type=str, default='default',
                        choices=['default', 'ytvis', 'spring'],
                        help='Specify dataset directory and file structure format.')
    parser.add_argument('--video_name', type=str, default=None,
                        help='Name of video folder to process (will process only this one)')
    
    default_query =""" Video: <image> You are an excellent video frame analyst.Given the video in the form of a sequence of frames above, describe the object in the masked region in the video.

        Follow these steps to analyze the masked object and generate the caption:

        1. Behavior Synthesis  
        - Summarize the object’s overall motion: note directionality, key interactions with the environment or other objects, and any accelerations or decelerations.  
        - Identify transitions in its activity (e.g., from stationary to moving, from linear to curved path).

        2. Visual-Physical Decomposition  
        - Describe the object’s shape, coloration, and texture as observed within the mask (e.g., “rectangular metal form with matte finish, dark gray”).  
        - Note any material cues or structural forms visible (e.g., rigid versus flexible, smooth versus ridged).  
        - Indicate whether these visual or physical attributes remain constant or vary across frames.

        3. Temporal Transition Tracking  
        - Systematically trace changes in the object’s appearance or pose over time (e.g., “tilts forward as it advances,” “surface catches highlight when rotating”).  
        - Highlight observable shifts in size, orientation, or occlusion that affect how the object is perceived.

        Constraints and Requirements  
        - Express your findings in a single coherent narrative without numbered lists or bullet points.  
        - State only verifiable facts—do not speculate or use metaphors, personification, or rhetorical flourishes.  
        - Do not reference frame indices or timestamps.  
        - Keep the description concise and precise, containing only information directly observable within the masked region.

        Final Step  
        Compose the final caption as one fluent and concise paragraph that objectively describes the masked object’s visual features, physical properties, motion patterns, and temporal changes throughout the video.
    """

    parser.add_argument('--query', type=str, default=default_query, help='Prompt for the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt_mode', type=str, default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str, default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p sampling')
    parser.add_argument('--output_image_dir', type=str, default=None, help='Directory to save output images with contours')
    parser.add_argument('--no_stream', action='store_true', help='Disable streaming output')
    parser.add_argument('--timing_log_path', type=str, default='timing_log.csv', help='CSV file path for output timing log')

    args = parser.parse_args()
    
    video_source_dir = args.input_dir
    videos = os.listdir(video_source_dir)
    os.makedirs(args.caption_path, exist_ok=True)

    disable_torch_init()
    prompt_modes = {"focal_prompt": "full+focal_crop"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DAM model...")
    model_load_start = time.time()
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode),
    ).to(device)
    model_load_end = time.time()
    print(f"Model loading took {model_load_end - model_load_start:.2f} seconds.")

    if args.video_name is not None:
        if args.video_name not in videos:
            raise ValueError(f"Specified video_name '{args.video_name}' not found in {video_source_dir}")
        videos = [args.video_name]

    timing_log_exists = os.path.isfile(args.timing_log_path)
    with open(args.timing_log_path, mode='a', newline='', encoding='utf-8') as timing_file:
        timing_writer = csv.writer(timing_file)
        if not timing_log_exists:
            timing_writer.writerow(['video_name', 'processing_time_seconds', 'timestamp'])

        total_start_time = time.time()
        
        for video in videos:
            video_path = os.path.join(video_source_dir, video)
            # Skip non-directory files
            if not os.path.isdir(video_path):
                continue

            video_start_time = time.time()
            video_name_without_ext = video
            print(f"\n{'='*20}\nProcessing: {video_name_without_ext}\n{'='*20}")
            
            # Dynamically determine frame and mask subpaths based on dataset format
            frame_subpath = ''
            mask_subpath = ''
            if args.dataset_format == 'ytvis':
                frame_subpath = 'JPEGImages'
                mask_subpath = 'Label'
            elif args.dataset_format == 'spring':
                frame_subpath = os.path.join('segmentation','frames','original')
                mask_subpath = os.path.join('segmentation','frames','masks')
            
            # Construct complete frame and mask directory paths
            if args.dataset_format == 'ytvis':
                frame_folder_path = os.path.join(video_source_dir, video_name_without_ext, 'rgb')
                mask_dir_for_video = os.path.join(video_source_dir, video_name_without_ext, 'Label')
            else: # For spring and default
                frame_folder_path = os.path.join(video_path, frame_subpath) if frame_subpath else video_path
                mask_dir_for_video = os.path.join(video_path, mask_subpath) if mask_subpath else video_path
            
            csv_path = os.path.join(args.caption_path, video_name_without_ext + '.csv')
            
            image_files = sorted(
                glob.glob(os.path.join(frame_folder_path, '*.jpg')) +
                glob.glob(os.path.join(frame_folder_path, '*.jpeg')) +
                glob.glob(os.path.join(frame_folder_path, '*.png'))
            )
            if not image_files:
                print(f"Warning: No image frames found in {frame_folder_path}. Skipping.")
                continue
            print(f"Found {len(image_files)} frames in {frame_folder_path}")
            
            raw_masks = load_instance_masks(image_files, mask_dir_for_video, dataset_format=args.dataset_format)
            
            with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['instance_id', 'description'])

                all_ids = set()
                for mask in raw_masks:
                    all_ids.update(np.unique(mask))
                instance_ids = sorted([i for i in all_ids if i != 0])

                for inst_id in instance_ids:
                    print(f"\n==== Instance {inst_id} ====")
                    masks = [(raw_mask == inst_id) for raw_mask in raw_masks]
                    visible_indices = [i for i, m in enumerate(masks) if np.any(m)]

                    if len(visible_indices) == 0:
                        print(f"Instance {inst_id} is not visible in any frame. Skipping.")
                        continue

                    if len(visible_indices) > 8:
                        sampled_indices = np.linspace(0, len(visible_indices)-1, 8, dtype=int)
                        selected_indices = [visible_indices[i] for i in sampled_indices]
                    else:
                        selected_indices = visible_indices

                    selected_files = [image_files[i] for i in selected_indices]
                    selected_masks = [masks[i] for i in selected_indices]

                    processed_images = [Image.open(f).convert('RGB') for f in selected_files]
                    processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in selected_masks]

                    num_images = len(processed_images)
                    image_placeholders = "<image>" * num_images
                    
                    if "You are an excellent" in args.query:
                        base_query_text = "You are an excellent" + args.query.split("You are an excellent")[1]
                    else:
                        base_query_text = "Describe the object in the masked region in the video."

                    dynamic_query = f"Video: {image_placeholders} {base_query_text}"

                    print(f"Generating description for instance {inst_id}:")
                    description = ''
                    if not args.no_stream:
                        for token in dam.get_description(
                                processed_images, processed_masks, dynamic_query,
                                streaming=True, temperature=args.temperature,
                                top_p=args.top_p, num_beams=1, max_new_tokens=256):
                            print_streaming(token)
                            description += token
                        print()
                    else:
                        description = dam.get_description(
                            processed_images, processed_masks, dynamic_query,
                            temperature=args.temperature, top_p=args.top_p,
                            num_beams=1, max_new_tokens=256)
                        print(description)

                    writer.writerow([inst_id, description.replace('\n', ' ')])

                    if args.output_image_dir:
                        inst_dir = os.path.join(args.output_image_dir, video_name_without_ext, f'instance_{inst_id}')
                        os.makedirs(inst_dir, exist_ok=True)
                        for idx, (img_path, mask) in enumerate(zip(selected_files, selected_masks)):
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img_np = img.astype(float) / 255.0
                            img_with_contour_np = add_contour(img_np, mask)
                            img_with_contour = Image.fromarray((img_with_contour_np * 255.0).astype(np.uint8))
                            output_path = os.path.join(inst_dir, f'frame_{selected_indices[idx]:05d}.png')
                            img_with_contour.save(output_path)
                        print(f"Output images for instance {inst_id} saved to {inst_dir}")
                
            print(f"\nDescriptions for {video_name_without_ext} saved to {csv_path}")

            video_end_time = time.time()
            processing_time = video_end_time - video_start_time
            print(f"Processing {video} took: {processing_time:.2f} seconds.")
            timing_writer.writerow([video, f"{processing_time:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])
            
        total_end_time = time.time()
        print(f"\n{'='*20}\nProcessing all videos took: {total_end_time - total_start_time:.2f} seconds.")
        print(f"Timing log saved to {args.timing_log_path}")