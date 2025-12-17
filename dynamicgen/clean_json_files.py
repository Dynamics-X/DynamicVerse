#!/usr/bin/env python3
"""
Clean JSON files in the analysis folder under scene folders
Keep only dynamic and reasoning keys, remove metadata
"""

import os
import json
from pathlib import Path
import argparse

def clean_json_files(base_dir):
    """Clean JSON files, keep only dynamic and reasoning fields"""
    
    # Define paths
    base_dir = Path(base_dir)
    
    processed_count = 0
    error_count = 0
    
    print("Starting to clean JSON files...")
    
    # Traverse all scene folders
    for scene_dir in sorted(base_dir.iterdir()):
        if scene_dir.is_dir() and scene_dir.name != "qvq_analysis_results":
            analysis_dir = scene_dir / "analysis"
            
            # Check if analysis folder exists
            if not analysis_dir.exists():
                continue
            
            # Find JSON files
            json_files = list(analysis_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    # Read original JSON file
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Create new dictionary with only dynamic and reasoning
                    cleaned_data = {}
                    
                    if 'dynamic' in data:
                        cleaned_data['dynamic'] = data['dynamic']
                    
                    if 'reasoning' in data:
                        cleaned_data['reasoning'] = data['reasoning']
                    
                    # Write back to file
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Cleaned: {json_file.relative_to(base_dir)}")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing file {json_file.relative_to(base_dir)}: {e}")
                    error_count += 1
    
    print(f"\nCleaning completed!")
    print(f"Successfully processed: {processed_count} files")
    if error_count > 0:
        print(f"Processing failed: {error_count} files")
    
    # Show examples of cleaned files
    print(f"\nExamples of cleaned file structure:")
    sample_files = []
    for scene_dir in sorted(base_dir.iterdir()):
        if scene_dir.is_dir() and scene_dir.name != "qvq_analysis_results":
            analysis_dir = scene_dir / "analysis"
            if analysis_dir.exists():
                json_files = list(analysis_dir.glob("*.json"))
                if json_files:
                    sample_files.append(json_files[0])
                    if len(sample_files) >= 3:  # Show only first 3 examples
                        break
    
    for sample_file in sample_files:
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            print(f"\n{sample_file.relative_to(base_dir)}:")
            print(f"  Fields included: {list(sample_data.keys())}")
            if 'dynamic' in sample_data:
                print(f"  Number of dynamic objects: {len(sample_data['dynamic'])}")
        except Exception as e:
            print(f"Error reading example file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize QVQ analysis files to corresponding scene directory analysis subfolders")
    parser.add_argument("--base_dir", type=str, help="Dataset root directory path, e.g. /path/to/dynpose_100k/test_data_inference")
    args = parser.parse_args()
    clean_json_files(args.base_dir) 