#!/bin/bash
# ================= Environment Variables and Parameter Settings =================
export DASHSCOPE_API_KEY=your_api_key_here
export CUDA_VISIBLE_DEVICES=0
# source ~/.bashrc
conda activate dynamicverse
DATA_ROOT="../../../data"
# DATA_ROOT="/path/to/alternative/datasets"
DATASET_NAME="demo"
SPLIT_PART="$1"
# SPLIT_PART="train_000" # Dynamically set by $1

DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}/${SPLIT_PART}"
KEY_FRAME_DATASET_NAME="${DATASET_NAME}/${SPLIT_PART}"
KEY_FRAME_DATASET_PATH="${DATA_ROOT}/key_frames/${KEY_FRAME_DATASET_NAME}"

echo "Original dataset name: ${DATASET_NAME}"
echo "Original dataset path: ${DATASET_PATH}"
echo "key frameDataset name: ${KEY_FRAME_DATASET_NAME}"
echo "key frameDataset path: ${KEY_FRAME_DATASET_PATH}"
echo "------------------------------------------------"

# ================= Motion-aware Key Frame extraction =================
echo "============= Starting to process dataset ${DATASET_NAME}'s key frame extraction ============="
start_time=$(date +%s)
python motion_aware_key_frame_extract.py \
    --input_root "${DATASET_PATH}" \
    --output_root "${KEY_FRAME_DATASET_PATH}" \
    --flow_model 'unimatch' 
end_time=$(date +%s)
echo "Key frame extraction time elapsed: $((end_time - start_time)) seconds"
echo ""

# ================= QVQ Analyzing =================
echo "============= Starting to process key frame dataset ${KEY_FRAME_DATASET_NAME}'s QVQ Analyzing ============="
start_time=$(date +%s)
cd ..
python batch_process_scene_caption.py \
    --data_root "${KEY_FRAME_DATASET_PATH}" \
    --max_frames 25
end_time=$(date +%s)
echo "QVQ Analyzing time elapsed: $((end_time - start_time)) seconds"
echo ""
