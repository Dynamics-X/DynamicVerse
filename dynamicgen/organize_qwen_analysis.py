#!/usr/bin/env python3
"""
Reorganize QVQ analysis result files
Move JSON files from qvq_analysis_results folder to corresponding scene folder's analysis subfolder
"""

import os
import shutil
import json
from pathlib import Path
import argparse

def organize_qvq_analysis_files(base_dir, target_dir):
    """Reorganize QVQ analysis result files"""
    
    # Define paths
    base_dir = Path(base_dir)
    target_dir = Path(target_dir)
    qvq_results_dir = base_dir
    
    # Ensure qvq_analysis_results folder exists
    if not qvq_results_dir.exists():
        print(f"Error: {qvq_results_dir} does not exist")
        return
    
    # Recursively get all QVQ analysis JSON files
    qvq_files = list(qvq_results_dir.rglob("*_qvq_analysis.json"))
    
    print(f"Found {len(qvq_files)} QVQ analysis files")
    
    for qvq_file in qvq_files:
        # Extract scene name from filename
        scene_name = qvq_file.stem.replace("_qvq_analysis", "")
        
        # Skip summary file
        if scene_name == "sintel_qvq_analysis_summary":
            continue
            
        scene_dir = target_dir / scene_name
        
        # Check if corresponding scene folder exists
        if not scene_dir.exists():
            print(f"Warning: Scene folder {scene_dir} does not exist, skipping {qvq_file.name}")
            continue
        
        # Create analysis folder
        analysis_dir = scene_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Define new filename
        new_filename = f"dynamic_objects_{scene_name}.json"
        new_file_path = analysis_dir / new_filename
        
        # Copy file
        try:
            shutil.copy2(qvq_file, new_file_path)
            print(f"Successfully copied: {qvq_file.name} -> {new_file_path.relative_to(target_dir)}")
        except Exception as e:
            print(f"Error copying file {qvq_file.name}: {e}")
    
    print("\nFile organization completed!")
    
    # Show result statistics
    print("\nAnalysis folder contents for each scene:")
    for scene_dir in sorted(target_dir.iterdir()):
        if scene_dir.is_dir():
            analysis_dir = scene_dir / "analysis"
            if analysis_dir.exists():
                files = list(analysis_dir.glob("*.json"))
                if files:
                    print(f"  {scene_dir.name}/analysis/: {len(files)} JSON files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize QVQ analysis files to corresponding scene directory analysis subfolders")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory path containing qvq_analysis.json files")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory path containing scene folders")
    args = parser.parse_args()
    
    organize_qvq_analysis_files(args.base_dir, args.target_dir)