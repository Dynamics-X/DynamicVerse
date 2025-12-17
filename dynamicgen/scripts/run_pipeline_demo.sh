#!/bin/bash
#  bash ./scripts_mask/run_pipeline.sh split_4 --qwen
# ================= Help and Usage =================
print_usage() {
    echo "Usage: $0 <split_part> [options]"
    echo ""
    echo "This is a data processing pipeline script."
    echo ""
    echo "Optional Stages (Options):"
    echo "  --key_frame        Run 1. Motion-aware Key Frame extraction"
    echo "  --qwen             Run 2. qwen + sa2va (batch_process_qwen_pipeline.py)"
    echo "  --sa2va            Run 3. SA2VA post-processing (organize/clean/copy)"
    echo "  --cotracker        Run 4. DynamicBA - Cotracker preprocessing"
    echo "  --unidepth         Run 5. DynamicBA - UniDepth preprocessing"
    echo "  --dynamicBA        Run 6. DynamicBA - optimization step"
    echo "  --all              Run all stages (1-6)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Default Behavior:"
    echo "  If no optional stages are specified, the script will default to run '--all'."
    echo ""
    echo "Examples:"
    echo "  # Run all stages"
    echo "  $0 train_000"
    echo "  # Run only key_frame and qwen"
    echo "  $0 train_000 --key_frame --qwen"
    echo "  # Run only DynamicBA last three steps"
    echo "  $0 train_000 --cotracker --unidepth --dynamicBA"
}

# ================= Parameter Parsing =================

# Check if help is needed
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_usage
    exit 0
fi

SPLIT_PART="$1"
shift # Consume the first parameter (split_part), remaining are optional flags

# Default flags are false
run_key_frame=false
run_qwen=false
run_sa2va=false
run_cotracker=false
run_unidepth=false
run_dynamicBA=false
run_all=false

# Check if optional flags are provided
if [ $# -eq 0 ]; then
    # If no flags are provided, default to --all
    echo "No stages specified, defaulting to run all stages..."
    run_all=true
else
    # Parse provided flags
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --key_frame)    run_key_frame=true; shift ;;
            --qwen)         run_qwen=true; shift ;;
            --sa2va)        run_sa2va=true; shift ;;
            --cotracker)    run_cotracker=true; shift ;;
            --unidepth)     run_unidepth=true; shift ;;
            --dynamicBA)    run_dynamicBA=true; shift ;;
            --all)          run_all=true; shift ;;
            *)
                echo "Error: Unknown option $1"
                print_usage
                exit 1
                ;;
        esac
    done
fi

# If specified --all，then override all flags to true
if [ "$run_all" = true ]; then
    run_key_frame=true
    run_qwen=true
    run_sa2va=true
    run_cotracker=true
    run_unidepth=true
    run_dynamicBA=true
fi


# ================= Environment Variables and Parameter Settings =================
export DASHSCOPE_API_KEY=your_api_key_here
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
conda activate dynamicverse

DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../data" && pwd)"
DATASET_NAME="demo"
# SPLIT_PART="train_000" # Dynamically set by $1

DATASET_PATH="${DATA_ROOT}/${DATASET_NAME}/${SPLIT_PART}"
KEY_FRAME_DATASET_NAME="${DATASET_NAME}/${SPLIT_PART}"
KEY_FRAME_DATASET_PATH="${DATA_ROOT}/key_frames/${KEY_FRAME_DATASET_NAME}"

echo "Original dataset name: ${DATASET_NAME}"
echo "Original dataset path: ${DATASET_PATH}"
echo "key frameDataset name: ${KEY_FRAME_DATASET_NAME}"
echo "key frameDataset path: ${KEY_FRAME_DATASET_PATH}"
echo "------------------------------------------------"


# ================= 1. Motion-aware Key Frame extraction =================
if [ "$run_key_frame" = true ]; then
    echo "============= [1/6] Starting to process dataset ${DATASET_NAME}'s key frame extraction ============="
    start_time=$(date +%s)
    python motion_aware_key_frame_extract.py \
        --input_root "${DATASET_PATH}" \
        --output_root "${KEY_FRAME_DATASET_PATH}" \
        --flow_model 'unimatch' \
        --sample_ratio 0.2
    end_time=$(date +%s)
    echo "Key frame extraction time elapsed: $((end_time - start_time)) seconds"
    echo ""
else
    echo "============= [1/6] Skipping Key Frame extraction ============="
    echo ""
fi


# ================= 2. qwen + sa2va =================
if [ "$run_qwen" = true ]; then
    echo "============= [2/6] Starting to run batch_process_qwen_api.py ============="
    start_time=$(date +%s)
    
    # Execute in parent directory
    python batch_process_qwen_pipeline.py \
        "${DATASET_PATH}" \
        "${DATASET_PATH}" \
        --base_frame_dir "${DATASET_PATH}" \
        --key_frame_dir "${KEY_FRAME_DATASET_PATH}" \
        # --skip_stage2
    end_time=$(date +%s)
    echo "Qwen API Processing time elapsed: $((end_time - start_time)) seconds"
    echo ""
else
    echo "============= [2/6] Skipping Qwen API Processing ============="
    echo ""
fi


# ================= 3. SA2VA post-processing (organize/clean/copy) =================
if [ "$run_sa2va" = true ]; then
    echo "============= [3/6] Starting SA2VA post-processing (organize/clean/copy) ============="
    echo "============= [3a] Starting directory format processing ============="
    start_time=$(date +%s)
    python organize_qwen_analysis.py --base_dir "${KEY_FRAME_DATASET_PATH}" --target_dir "${DATASET_PATH}"
    end_time=$(date +%s)
    echo "Directory organize time elapsed: $((end_time - start_time)) seconds"
    echo ""

    # ================= Clean redundant JSON files =================
    echo "============= [3b] Starting cleanup JSON files ============="
    start_time=$(date +%s)
    python clean_json_files.py --base_dir "${DATASET_PATH}"
    end_time=$(date +%s)
    echo "JSON Cleanup time: $((end_time - start_time)) seconds"
    echo ""

    # ================= Process directories for each scene =================
    echo "============= [3c] Starting to process directories for each scene (copy masks) ============="
    start_time=$(date +%s)

    # Traverse all scenes under DATASET_PATH
    for scene_path in "${DATASET_PATH}"/*; do
        if [ -d "${scene_path}" ]; then
            scene_name=$(basename "${scene_path}")
            echo "Processing scene: ${scene_name}"

            # Create target directory
            target_dir="${scene_path}/qwen/Annotations"
            mkdir -p "${target_dir}"

            # Copy mask files
            mask_dir="${scene_path}/segmentation/frames/masks"
            if [ -d "${mask_dir}" ]; then
                cp "${mask_dir}"/*.png "${target_dir}/"
                echo "Copied masks from ${mask_dir} to ${target_dir}"
            else
                echo "Warning: Mask directory not found in scene ${scene_name}: ${mask_dir}"
            fi
            echo ""
        fi
    done
    
    end_time=$(date +%s)
    echo "Scene directory processing time: $((end_time - start_time)) seconds"
    echo ""
else
    echo "============= [3/6] Skipping SA2VA post-processing ============="
    echo ""
fi


# ================= 4, 5, 6. DynamicBA preprocessing and optimization =================
# (These steps share the same working directory)
if [ "$run_cotracker" = true ] || [ "$run_unidepth" = true ] || [ "$run_dynamicBA" = true ]; then
    cd ..
    echo "Current working directory: $(pwd)"
    echo ""

    # ================= 4. Cotracker =================
    if [ "$run_cotracker" = true ]; then
        echo "============= [4/6] Starting to run Cotracker ============="
        start_time=$(date +%s)
        python ./preprocess/run_cotracker.py --workdir "$DATASET_PATH" --interval 10 --grid_size 75
        end_time=$(date +%s)
        echo "Cotracker time elapsed: $((end_time - start_time)) seconds"
        echo ""
    else
        echo "============= [4/6] Skipping Cotracker ============="
        echo ""
    fi

    # ================= 5. UniDepth =================
    if [ "$run_unidepth" = true ]; then
        echo "============= [5/6] Starting to run UniDepth ============="
        start_time=$(date +%s)
        python ./preprocess/run_unidepth.py --workdir "$DATASET_PATH"
        end_time=$(date +%s)
        echo "UniDepth time elapsed: $((end_time - start_time)) seconds"
        echo ""
    else
        echo "============= [5/6] Skipping UniDepth ============="
        echo ""
    fi

    # ================= 6. dynamicBA optimization =================
    if [ "$run_dynamicBA" = true ]; then
        echo "============= [6/6] Starting dynamicBA optimization ============="
        start_time=$(date +%s)
        python ./dynamicBA/run.py   --config ./dynamicBA/config/config.yaml \
                                    --workdir "$DATASET_PATH" \
                                    --max_frames 120
        end_time=$(date +%s)
        echo "DynamicBA optimization time elapsed: $((end_time - start_time)) seconds"
        echo ""
    else
        echo "============= [6/6] Skipping DynamicBA optimization ============="
        echo ""
    fi

else
    echo "============= [4-6/6] Skipping all DynamicBA related steps ============="
    echo ""
fi

echo "============= All selected tasks completed ============="
