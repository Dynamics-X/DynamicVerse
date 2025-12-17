#!/usr/bin/env python3
"""
# MODIFIED: Batch process all scenes in dataset, generate captions for each scene's rgb folder
# This is a single script that integrates qwen_keyframes_analysis.py functionality.
"""

import os
import json
import argparse
import time
from pathlib import Path
import glob
import tempfile  # <-- 1. Core solution: import tempfile
from datetime import datetime

# --- Imports from qwen_keyframes_analysis (MERGED) ---
from openai import OpenAI
import base64
import math
import gc
import numpy as np
from PIL import Image
# --- End of merged import section ---


# ====================================================================
# Functions from qwen_keyframes_analysis.py (MERGED)
# ====================================================================

def encode_image(image_path):
    """Encode image file to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_keyframes_from_directory(keyframes_dir, max_frames=64):
    """
    Load keyframes that have been extracted from directory
    
    Args:
        keyframes_dir: Keyframe directory path
        max_frames: Maximum number of frames to use, avoid large grids
    
    Returns:
        frames: List of PIL Image objects
        frame_info: List of frame information
    """
    print(f"Loading keyframes from directory: {keyframes_dir}")
    
    # Find PNG image files
    image_patterns = [
        os.path.join(keyframes_dir, "*.png"),
        os.path.join(keyframes_dir, "*.jpg"),
        os.path.join(keyframes_dir, "*.jpeg")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
    
    # Sort by filename
    image_paths.sort()
    
    if not image_paths:
        raise ValueError(f"No image files found in directory {keyframes_dir}")
    
    print(f"Found {len(image_paths)} image files")
    
    # If there are too many images, perform uniform sampling
    if len(image_paths) > max_frames:
        print(f"Number of images ({len(image_paths)}) exceeds maximum limit ({max_frames}), performing uniform sampling")
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]
        print(f"After sampling, using {len(image_paths)} images")
    
    # Load images
    frames = []
    frame_info = []
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            frames.append(img)
            
            # Extract frame information (from filename)
            filename = os.path.basename(image_path)
            frame_info.append({
                'index': i,
                'filename': filename,
                'path': image_path
            })
            
        except Exception as e:
            print(f"Warning: Unable to load image {image_path}: {e}")
            continue
    
    if not frames:
        raise ValueError("No images loaded successfully")
    
    print(f"Successfully loaded {len(frames)} keyframes")
    return frames, frame_info

def call_qvq_api_multi_images(frames, prompt):
    """
    Call QVQ API for multi-image analysis
    
    Args:
        frames: List of PIL Image objects
        prompt: Analysis prompt
    
    Returns:
        response: API response text
    """
    # 2. MODIFIED: No longer need os.makedirs and temp_dir parameters
    
    # Save all images and encode
    content_list = []
    
    try:
        print(f"Processing {len(frames)} keyframes...")
        
        # 3. MODIFIED: Use tempfile.TemporaryDirectory to create unique temporary directory
        with tempfile.TemporaryDirectory(prefix="qvq_batch_") as temp_dir:
            print(f"Using independent temporary directory: {temp_dir}")
            
            # Create temporary files for each image and encode
            for i, frame in enumerate(frames):
                # 4. MODIFIED: Create file in independent temporary directory
                temp_image_path = os.path.join(temp_dir, f"keyframe_{i:03d}.jpg")
                
                # Save image
                frame.save(temp_image_path, quality=95)
                
                # Encode image
                base64_image = encode_image(temp_image_path)
                
                # Add to content list
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                })
                
                if (i + 1) % 10 == 0:  # Display progress every 10 images
                    print(f"Processed {i + 1}/{len(frames)} keyframes")
        
            # 5. MODIFIED: 'with' block ends, temporary directory and all images deleted automatically
            # 'content_list' already contains all base64 data, no longer needs local files
            
            # Add text prompt
            content_list.append({"type": "text", "text": prompt})
            
            print(f"All {len(frames)} keyframes processed successfully")        # Get API key and verify
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set or empty")
        
        print(f"API Key status: {'✅ Set' if api_key else '❌ Not set'}")
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            # base_url="http://localhost:11434/v1", 
            # api_key="ollama"
            # base_url="http://127.0.0.1:22002/v1",
            # api_key="none"
        )
        
        print("Calling QVQ API...")
        
        # Create chat completion request
        completion = client.chat.completions.create(
            model="qvq-max-latest",
            # model="qwen2.5vl:72b",
            # model="Qwen2.5-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ],
            stream=True,
        )
        
        reasoning_content = ""
        answer_content = ""
        is_answering = False
        
        print("\n" + "=" * 30 + " QVQ Reasoning Process " + "=" * 30)
        
        for chunk in completion:
            if not chunk.choices:
                if hasattr(chunk, 'usage'):
                    print(f"\nAPI usage: {chunk.usage}")
            else:
                delta = chunk.choices[0].delta
                
                # Processing reasoning process
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # Starting to output final answer
                    if delta.content and delta.content != "" and not is_answering:
                        print("\n" + "=" * 30 + " QVQ Final Answer " + "=" * 30)
                        is_answering = True
                    
                    # Processing final answer
                    if delta.content:
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content
        
        print("\n" + "=" * 80)
        return answer_content
    
    finally:
        # 6. MODIFIED: Remove all manual file cleaning code
        # tempfile handles it automatically
        
        # Clean memory (this is still retained)
        print("\n🧹 Cleaning memory...")
        gc.collect()
        print("✅ Memory cleaning completed")

def parse_json_from_response(response_text):
    """
    Extract and parse JSON object from response text
    
    Args:
        response_text: API response text
    
    Returns:
        parsed_json: Parsed JSON object
    """
    try:
        # Try to parse the entire response directly
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If failed, look for JSON pattern
        import re
        
        # Look for JSON object pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                result = json.loads(match.group(0))
                print("✅ Successfully parsed JSON object from response")
                return result
            except json.JSONDecodeError:
                continue
        
        # If all failed, return default structure
        print("⚠️ Could not parse valid JSON, returning default structure")
        return {
            "dynamic": [],
            "reasoning": {},
            "raw_response": response_text
        }

def validate_json_format(data):
    """
    Validate JSON format compliance
    
    Args:
        data: Parsed JSON data
    
    Returns:
        is_valid: Whether valid
    """
    if not isinstance(data, dict):
        return False
    
    # Check necessary keys
    if "dynamic" not in data or "reasoning" not in data:
        return False
    
    # Check data types
    if not isinstance(data.get("dynamic"), list):
        return False
    
    if not isinstance(data.get("reasoning"), dict):
        return False
    
    return True

# ====================================================================
# Functions from batch_qwen_scene_analysis.py
# ====================================================================

def find_all_scene_rgb_directories(data_root):
    """
    Find all scene rgb subdirectories
    
    Args:
        data_root: Data root directory
    
    Returns:
        scene_dirs: [(scene_name, rgb_dir_path), ...]
    """
    scene_dirs = []
    
    # Traverse data root directory
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        
        # Check if directory
        if os.path.isdir(item_path):
            # Check if rgb subdirectory exists
            rgb_dir = os.path.join(item_path, "rgb")
            if os.path.exists(rgb_dir) and os.path.isdir(rgb_dir):
                # Check if rgb directory contains image files
                image_files = []
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
                
                if image_files:
                    scene_dirs.append((item, rgb_dir))
                    print(f"✅ Found scene: {item} (contains {len(image_files)} images)")
                else:
                    print(f"⚠️ Skipping scene: {item} (rgb directory is empty)")
            else:
                print(f"⚠️ Skipping scene: {item} (no rgb directory)")
    
    # Sort by scene name
    scene_dirs.sort(key=lambda x: x[0])
    
    print(f"\nTotal found {len(scene_dirs)} valid scenes")
    return scene_dirs

def analyze_single_scene(scene_name, scene_path, rgb_dir, max_frames=64, temp_dir="temp_qvq"):
    """
    # MODIFIED: Analyze single scene's rgb folder and generate scene description
    # (Note: temp_dir parameter is now redundant, but kept for function signature consistency)
    
    Args:
        scene_name: Scene name
        scene_path: Scene path
        rgb_dir: RGB folder path
        max_frames: Maximum number of frames
        temp_dir: (Deprecated) Temporary file directory
    
    Returns:
        success: Whether successful
        result_path: Result file path
    """
    print(f"\n🔄 Analyzing scene: {scene_name}")
    print(f"📁 RGB directory: {rgb_dir}")
    start_time = time.time()
    try:
        # 1. Load keyframes
        print("🔄 Loading keyframes...")
        # (Function is now defined in this file)
        frames, frame_info = load_keyframes_from_directory(rgb_dir, max_frames=max_frames)
        
        # 2. Call API
        print("🔄 Calling API to generate scene description...")
        
        # CHANGED: Replace with new scene description prompt
        # ... (prompt content remains unchanged)
        caption_prompt = """
            You are a spatial-semantic video analyst specializing in reconstructing dynamic scene graphs. Your task is to analyze the scene, parse object hierarchies, quantify spatial relationships, and trace motion patterns to build a concise and relevant caption for the video.

            Follow these steps to analyze the scene and generate the caption:

            1. Scene Context Decomposition
            Determine whether the scene is indoor, outdoor, or a functional space.
            Describe the background by noting the lighting conditions (e.g., bright, dim, natural light) and the dominant environmental elements (e.g., trees, buildings, furniture).
            Calculate the number of objects in each hierarchical tier:
                - Tier 1: Primary interactive agents (e.g., humans, vehicles)
                - Tier 2: Functional complements (e.g., tools, furniture)
                - Tier 3: Static environment (e.g., walls, terrain)

            2. Object Hierarchy Categorization**
            For each object in the scene, analyze:
                - Its class or semantic label (e.g., "coffee mug", "chair", "car")
                - Its location or spatial position within the entire scene (e.g., "on the table", "in the corner", "near the door")
                - Its size relative to the scene (e.g., "small", "medium", "large")
                - Its orientation or cardinal/relative facing direction (e.g., "facing north", "turned towards the window")

            3. Spatial Relationship Mapping
            Define scene graph edges by:
                - Assessing proximity: Determine if objects are adjacent, nearby, or distant from each other
                - Identifying directional relationships: Note positions such as left-right, front-behind, or above-below
                - Recognizing composite groups: Identify clustered objects (e.g., "dishwasher containing plates")
            Highlight 2-3 critical relationships shaping the scene's logic (e.g., a chair facing a desk indicating a workspace).

            4. Motion Pattern Analysis
            For dynamic objects, examine:
                - Trajectory patterns: Trace whether the motion is linear, curved, or zigzag
                - Speed: Observe if it is accelerating, decelerating, or constant
                - Interaction zones: Map areas where motion paths intersect or come close to each other

            5. Scenario Reasoning
            Synthesize your observations to infer:
                - Plausible human activities (e.g., food preparation in a kitchen)
                - Scene functionality (e.g., a workspace, a recreational area)
                - Aesthetic style, if observable (e.g., modern, rustic)
            Ensure all inferences strictly adhere to observable evidence from object placements, spatial relationships, and motion patterns.
            For instance, a bicycle moving toward an intersection suggests urban commuting, while a cluster of chairs facing a screen implies a meeting space.

            Constraints and Requirements
            - Use strict geometric terminology and avoid metaphorical descriptions.
            - Maintain object permanence by tracking identical objects across frames.
            - Disclose limitations, e.g., "Hand-object occlusion prevents interaction verification".
            - All inferences must derive exclusively from observable spatial-motion data.
            - The final caption should be concise, avoiding overly detailed or lengthy descriptions.
            - Ensure the description is highly relevant to the video content, excluding irrelevant information.
            - Synthesize findings into a single narrative without markdown.

            Final Step
            Synthesize your observations and inferences into a concise caption that describes the scene, its objects, their relationships, and any notable motion patterns or activities, adhering to the constraints and ensuring relevance to the video content.
            The final output must be a single, pure narrative paragraph, strictly adhering to the word count limit (approx. 100 words, max. 150 words) and containing no markdown or surrounding format.
"""
        max_retries = 3
        result_text = None
        
        for retry in range(max_retries):
            print(f"📡 Attempt {retry + 1}/{max_retries}...")
            
            try:
                # 7. MODIFIED: Call no longer needs temp_dir parameter
                response = call_qvq_api_multi_images(frames, caption_prompt)
                
                # MODIFIED: Directly get the text content from the response.
                # ... (subsequent logic remains unchanged)
                if response and isinstance(response, str):
                    result_text = response
                elif hasattr(response, 'output') and hasattr(response.output, 'text'): # A common structure for API responses
                    result_text = response.output.text
                else: # Fallback, assuming the response might be a dict
                    result_text = str(response)

                # MODIFIED: Validate the text response instead of JSON format
                if result_text and len(result_text) > 50:  # Simple validation: not empty and has substantial content
                    print("✅ Successfully generated valid scene description!")
                    break
                else:
                    print(f"❌ Returned description text invalid or too short: '{result_text}'")
                    result_text = None # Reset on failure
                    if retry < max_retries - 1:
                        print("🔄 Preparing retry...")
                    else:
                        print("⚠️ Reached maximum retries, unable to generate valid description")
                        
            except Exception as e:
                print(f"❌ API call failed: {e}")
                if retry < max_retries - 1:
                    print("🔄 Preparing retry...")
                else:
                    print("❌ Reached maximum retries, analysis failed")
                    return False, None
        
        if not result_text:
             return False, None

        # 3. Save results
        print("🔄 Saving analysis results...")
        
        # MODIFIED: Create a new result structure for the caption
        output_data = {
            "caption": result_text,
            "metadata": {
                "scene_name": scene_name,
                "rgb_dir": rgb_dir,
                "total_frames_used": len(frames),
                "analysis_method": "multi_image_sequence_captioning",
                "original_resolution_preserved": True,
                "frame_info": frame_info,
                "analysis_time": datetime.now().isoformat(),
                "inference_time": time.time() - start_time
            }
        }
        
        # MODIFIED: Change the result filename
        result_filename = f"{scene_name}_scene_caption_analysis.json"
        result_path = os.path.join(scene_path, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        end_time = time.time()
        print(f"✅ Analysis completed: {result_path}")
        print(f"⏱️ Analysis took: {end_time - start_time:.2f} seconds, frames: {len(frames)}")
        print(f"📋 Generated description (first 100 characters): \n   '{result_text[:100]}...'")
        
        return True, result_path
        
    except Exception as e:
        print(f"❌ Scene analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_summary_report(output_dir, results, skipped_count=0):
    """
    # MODIFIED: Create summary report, summarize captions
    
    Args:
        output_dir: Output directory
        results: Analysis result list
        skipped_count: Number of skipped scenes
    """
    print("\n🔄 Generating summary report...")
    
    summary = {
        "total_scenes": len(results),
        "successful_analyses": len([r for r in results if r["success"]]),
        "failed_analyses": len([r for r in results if not r["success"]]),
        "skipped_scenes": skipped_count,
        "newly_analyzed_scenes": len(results) - skipped_count,
        "analysis_time": datetime.now().isoformat(),
        "scene_results": []
    }
    
    # MODIFIED: Logic to summarize captions instead of dynamic objects
    for result in results:
        scene_summary = {
            "scene_name": result["scene_name"],
            "success": result["success"],
            "result_file": result.get("result_path", ""),
            "caption_preview": ""
        }
        
        if result["success"] and result.get("result_path"):
            try:
                with open(result["result_path"], 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                    caption = scene_data.get("caption", "")
                    scene_summary["caption_preview"] = caption[:150] + "..." if caption else ""
            except Exception as e:
                print(f"⚠️ Unable to read result file: {result['result_path']}: {e}")
        
        summary["scene_results"].append(scene_summary)
    
    # MODIFIED: Remove dynamic object counting
    
    # Save summary report
    summary_path = os.path.join(output_dir, "caption_analysis_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Summary report saved: {summary_path}")
    
    # Print summary information
    print("\n" + "=" * 60)
    print("📊 Analysis Summary")
    print("=" * 60)
    print(f"📁 Total scenes: {summary['total_scenes']}")
    print(f"⏭️ Skipped scenes: {summary['skipped_scenes']} (already analyzed)")
    print(f"🚀 Newly analyzed scenes: {summary['newly_analyzed_scenes']}")
    print(f"✅ Successful analyses: {summary['successful_analyses']}")
    print(f"❌ Failed analyses: {summary['failed_analyses']}")
    print("=" * 60)

# This function remains unchanged
def auto_detect_data_root():
    """Automatically detect data root directory"""
    possible_paths = [
        "./sintel_sampled_output", "./output", "./sintel_output",
        "./data/sintel_sampled_output", "../sintel_sampled_output",
        "./sintel_processed", "./processed_sintel"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            rgb_count = 0
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "rgb")):
                        rgb_count += 1
                if rgb_count > 0:
                    print(f"✅ Automatically detected data directory: {path} (contains {rgb_count} scenes)")
                    return path
            except:
                continue
    return None

def main():
    # MODIFIED: Update descriptions
    parser = argparse.ArgumentParser(description="Batch process all scenes in dataset, generate captions")
    parser.add_argument("--data_root", 
                        default="/data/workspace_hyz/projects/dynamicBA/data/ego_demo1_inference",
                        help="Dataset root directory containing scene subdirectories (leave empty for auto-detection)")
    parser.add_argument("--max_frames", type=int, default=25,
                        help="Maximum number of frames per scene (default: 25)")
    parser.add_argument("--temp_dir", default="temp_caption_batch",
                        help="Temporary file directory (default: temp_caption_batch) - [Note: This parameter is deprecated in the new version]")
    parser.add_argument("--scene_filter", type=str, default="",
                        help="Scene name filter (only analyze scenes containing this string)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay time between scenes (seconds, avoid API limits)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-analysis of all scenes, even if result files already exist")
    parser.add_argument("--output_dir", default="./caption_analysis_results",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    if not args.data_root:
        print("🔍 No data root specified, processing auto-detection...")
        args.data_root = auto_detect_data_root()
        if not args.data_root:
            print("❌ Could not auto-detect data directory, please specify with --data_root.")
            return
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY environment variable not set")
        print("Please set API key first: export DASHSCOPE_API_KEY=your_api_key_here")
        return
    else:
        print("✅ DASHSCOPE_API_KEY is set")
    
    print("=" * 80)
    print("🎬 Dataset batch scene description generation")
    print("=" * 80)
    print(f"📁 Data root directory: {args.data_root}")
    print(f"📄 Output directory: {args.output_dir}")
    print(f"🖼️ Maximum frames: {args.max_frames}")
    print(f"⏱️ Delay time: {args.delay} seconds")
    if args.scene_filter:
        print(f"🔍 Scene filter: '{args.scene_filter}'")
    if args.force:
        print(f"🔄 Force re-analysis: Yes (ignore existing results)")
    else:
        print(f"⏭️ Skip already analyzed scenes: Yes (use --force to force re-analysis)")
    print("=" * 80)
    
    try:
        if not os.path.exists(args.data_root):
            raise ValueError(f"Data root directory does not exist: {args.data_root}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("\n🔄 Step 1: Finding scene directories...")
        scene_dirs = find_all_scene_rgb_directories(args.data_root)
        
        if not scene_dirs:
            raise ValueError("No valid scene directories found")
        
        if args.scene_filter:
            original_count = len(scene_dirs)
            scene_dirs = [(name, path) for name, path in scene_dirs if args.scene_filter in name]
            print(f"🔍 Scene filter screening: {original_count} -> {len(scene_dirs)} scenes")
        
        print(f"\n🔄 Step 2: Batch analyzing {len(scene_dirs)} scenes...")
        
        results = []
        start_time = time.time()
        skipped_count = 0
        
        for i, (scene_name, rgb_dir) in enumerate(scene_dirs, 1):
            print(f"\n{'='*20} Scene {i}/{len(scene_dirs)} {'='*20}")
            
            # MODIFIED: Update result filename for checking existence
            result_filename = f"{scene_name}_scene_caption_analysis.json"
            scene_path = os.path.join(args.data_root, scene_name)
            result_path = os.path.join(scene_path, result_filename)
            
            if os.path.exists(result_path) and not args.force:
                print(f"✅ Scene {scene_name} already analyzed, skipping")
                print(f"📄 Result file: {result_path}")
                
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        existing_result = json.load(f)
                    
                    # MODIFIED: Check integrity for the new format
                    if "caption" in existing_result and existing_result["caption"]:
                        print(f"💬 Existing description complete")
                        results.append({
                            "scene_name": scene_name, "rgb_dir": rgb_dir,
                            "success": True, "result_path": result_path
                        })
                        skipped_count += 1
                        continue
                    else:
                        print(f"⚠️ Existing result file incomplete, re-analyzing...")
                        
                except Exception as e:
                    print(f"⚠️ Failed to read existing result: {e}, re-analyzing...")
            elif os.path.exists(result_path) and args.force:
                print(f"🔄 Using --force parameter, re-analyzing scene {scene_name}")
            
            success, result_path = analyze_single_scene(
                scene_name, scene_path, rgb_dir, 
                args.max_frames, args.temp_dir # 8. MODIFIED: args.temp_dir is still passed, but analyze_single_scene will ignore it
            )
            
            results.append({
                "scene_name": scene_name, "rgb_dir": rgb_dir,
                "success": success, "result_path": result_path
            })

            if i < len(scene_dirs) and args.delay > 0:
                print(f"⏱️ Waiting {args.delay} seconds...")
                time.sleep(args.delay)
        
        print(f"\n🔄 Step 3: Generating summary report...")
        create_summary_report(args.output_dir, results, skipped_count)
        
        total_time = time.time() - start_time
        successful_count = len([r for r in results if r["success"]])
        
        print(f"\n🎉 Batch analyzing completed!")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"📊 Processing statistics: {successful_count}/{len(scene_dirs)} scenes successful")
        print(f"📁 Results directory: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Batch analyzing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()