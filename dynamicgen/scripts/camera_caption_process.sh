export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python ../batch_process_camera_caption.py \
  --model 'qwen2.5-vl-7b' \
  --checkpoint '/path/to/model/checkpoints/qwen2.5-vl-7b-cam-motion' \
  --video_dir '/path/to/datasets/videos' \
  --output_dir scores/result/output_dataset/
  