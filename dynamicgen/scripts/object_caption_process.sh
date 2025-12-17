#!/bin/bash

# ==============================================================================
# Bash script: Automatic processing of "dynpose" dataset
#
# Usage instructions:
# 1. Place your Python script in the same directory as this Bash script, or modify the SCRIPT_NAME path.
# 2. Modify the "Configuration Section" below, fill in your own dataset, model, and result paths.
# 3. Grant execution permission to the script: chmod +x dynpose.sh
# 4. Execute the script: ./dynpose.sh
# ==============================================================================

# --- Configuration Section ---
# Please modify the paths here to your actual paths

# Your Python script filename
SCRIPT_NAME="../batch_process_object_caption.py"

# Root directory of the "spring" dataset, which contains all video ID folders (e.g., 0002, 0004)
DATASET_ROOT="/path/to/datasets/DAVIS"

# Directory where DAM model weights are located
MODEL_PATH="/path/to/model/checkpoints/DAM-3B-Video"

# Root directory for storing all output results
RESULTS_ROOT="/path/to/results/captions/object_captions/davis"


# --- Main Script Logic ---

# Check if Python script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: Python script '$SCRIPT_NAME' does not exist."
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: Dataset root directory '$DATASET_ROOT' does not exist."
    exit 1
fi

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' does not exist."
    exit 1
fi


echo ""
echo "================================================================"
echo ">>> Starting to process 'spring' dataset..."
echo ">>> Input directory: ${DATASET_ROOT}"
echo "================================================================"

# Execute Python script
# The script will automatically traverse all video folders under the input directory
python3 "$SCRIPT_NAME" \
    --input_dir "$DATASET_ROOT" \
    --caption_path "${RESULTS_ROOT}" \
    --model_path "$MODEL_PATH" \
    --dataset_format spring \
    --timing_log_path "${RESULTS_ROOT}/timing_log.csv"

echo ""
echo "================================================================"
echo ">>> 'spring' dataset processing completed! Results saved in ${RESULTS_ROOT}."
echo "================================================================"
