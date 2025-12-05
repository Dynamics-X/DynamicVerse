
<h2 align="center"> <a href="https://arxiv.org/abs/2512.03000">DynamicVerse: A Physically-Aware Multimodal Framework<br>for 4D World Modeling</a></h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-251203.03000-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.03000) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/kairunwen/DynamicVerse) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://dynamic-verse.github.io/) [![X](https://img.shields.io/badge/-Twitter@Kairun%20Wen%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/KairunWen)  [![youtube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=0h7XysIpG8Y)
</h5>

<br>

## Overview

DynamicVerse is an integrated framework for dynamic scene understanding and 4D reconstruction, combining advanced visual models such as Sa2VA, Qwen-VL, DAM, CameraBench, CoTracker, and UniDepth to achieve end-to-end processing from video to 4D scenes.

## Key Features

- 🎬 **Dynamic Scene Analysis**: Supports video keyframe extraction and motion-aware analysis
- 🔍 **Multimodal Understanding**: Integrates vision-language models for scene description and object recognition
- 🎯 **Dense Segmentation**: Precise object segmentation and tracking based on Sa2VA
- 📊 **4D Reconstruction**: Complete pipeline from video to 4D scene reconstruction

## Project Structure

```
DynamicVerse/
├── dynamicBA/           # 4D scene reconstruction module
│   ├── unimatch/        # Optical flow and depth estimation
│   ├── dataset_prepare/ # Data preprocessing tools
│   └── config/          # Configuration files
├── Sa2VA/               # Vision-language understanding module
│   ├── vlm/             # Vision-language models
│   └── third_parts/     # Third-party integrations
├── data/                # Dataset directory
├── scripts/             # Preprocessing scripts
└── dynamicgen/          # Pipeline execution
    └── qwen_analysis/   # Qwen analysis
```

## Installation

### 1. DynamicVerse Environment

```bash
git clone --recurse-submodules https://github.com/Dynamics-X/DynamicVerse.git 
cd DynamicVerse
conda create -n dynamicverse python=3.10
conda activate dynamicverse
bash scripts/install.sh
```

### 2. Download Pre-trained Models

```bash
bash scripts/download_weights.sh
```

This script will automatically download the following models:
- CoTracker3 (for motion tracking)
- UniDepth (for depth estimation)
- Sa2VA-8B (multimodal understanding model)
- Qwen2.5-VL-72B-Instruct (vision-language model)(optional)

## Quick Start

### Run DynamicGen Demo

Process a complete geometric scene pipeline:

```bash
cd dynamicgen
bash scripts/run_pipeline_demo.sh '' -all
```

This script executes the following steps:
1. **Keyframe Extraction**: Motion-aware video keyframe extraction
2. **Scene Analysis**: Multimodal analysis using Qwen and Sa2VA
3. **Segmentation Processing**: Generate object masks and organize output
4. **4D Reconstruction** (Optional): Complete 4D scene reconstruction using dynamicBA


### Qwen2.5-VL Configuration

Qwen2.5-VL can be used in two ways:

#### Option 1: API Service (Default)

For API service usage:

1. **Set API Key**: Set environment variable when running scripts
   ```bash
   export DASHSCOPE_API_KEY=your_api_key
   ```
   Or set it directly in `dynamicgen/scripts/run_pipeline_demo.sh` 

2. **Modify API Configuration**: Edit `dynamicgen/stage1_qwen.py` 
   ```python
   client = OpenAI(
       api_key=api_key,  # Use API key from environment variable
       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # API service address
   )

   model="qvq-max-latest"  # Or other Qwen models
   ```

#### Option 2: Local Deployment

For local deployment, modify `dynamicgen/stage1_qwen.py` to local service configuration:

```python
client = OpenAI(
    base_url="http://127.0.0.1:22002/v1",  # Local service address
    api_key="none"  # Not needed for local service
)

# Specify model name at line 213
model="Qwen/Qwen2.5-VL-72B-Instruct"
```

**Install Dependencies:**
```bash
pip install accelerate
pip install qwen-vl-utils==0.0.14
uv pip install -U vllm  # Requires vllm>=0.11.0
```

**Start Local Service:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model <ckpt_path> \
  --served-model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --host 0.0.0.0 \
  --port 22002 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --quantization fp8 \
  --distributed-executor-backend mp
```

For detailed deployment instructions, refer to [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file)


## Processing Pipeline

### 1. Data Preparation
Place videos or image sequences in the `data/` directory

### 2. Keyframe Extraction
```python
python motion_aware_key_frame_extract.py \
    --input_root <input_path> \
    --output_root <output_path> \
    --flow_model 'unimatch'
```

### 3. Multimodal Analysis
```python
python batch_process_qwen_pipeline.py \
    <dataset_path> \
    <output_path> \
    --base_frame_dir <base_frame_dir> \
    --key_frame_dir <key_frame_dir>
```

### 4. 4D Scene Reconstruction (Optional)
```python
cd dynamicBA
python ./dynamicBA/run.py \
    --config ./dynamicBA/config/config.yaml \
    --experiment_name base \
    --opt_intrinsics \
    --workdir <workdir>
```

## Output Directory Structure

After processing, the following directory structure is generated:

```
data/
├── key_frames/                  # Keyframe extraction results
│   └── <dataset_name>/         # Dataset name
│       └── <scene_id>/         # Scene ID
│           ├── frame_*.jpg
│           └── keyframe_info.json
└── demo/                        # Processed scene data
    └── <scene_id>/              # Scene ID directory
        ├── videos/              # Original video files
        │   └── <scene_id>.mp4
        ├── rgb/                 # Extracted RGB frames
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        ├── analysis/            # Scene analysis results
        │   └── dynamic_objects_<scene_id>.json  # Dynamic object detection results
        ├── qwen/                # Qwen model outputs
        │   └── Annotations/     # Segmentation annotations
        │       ├── frame_00000.png
        │       ├── frame_00001.png
        │       └── ...
        ├── segmentation/        # Sa2VA segmentation results
        │   ├── frames/          # Frame-level segmentation results
        │   │   ├── original/   # Original frames
        │   │   ├── masks/      # Segmentation masks
        │   │   ├── overlay/    # Overlay visualizations
        │   │   └── segmented/  # Segmented images
        │   ├── videos/          # Segmentation videos
        │   │   ├── original.mp4    # Original video
        │   │   ├── masks.mp4       # Mask video
        │   │   ├── overlay.mp4     # Overlay video
        │   │   └── segmented.mp4   # Segmented video
        │   ├── instance_labels.json    # Instance label information
        │   └── result_summary.json     # Segmentation result summary
        ├── dynamicBA/ (Optional)        # 4D reconstruction results
        │   ├── fused_4d.npz    # Fused 4D data
        │   ├── depth/          # Depth maps
        │   └── flow/           # Optical flow data
        └── processing_log_<scene_id>.log  # Processing log
```

### Output Files Description

- **dynamic_objects_*.json**: Contains detected dynamic object information, including position, category, and tracking ID
- **instance_labels.json**: Label mapping for each instance, used for multi-object segmentation
- **result_summary.json**: Segmentation result statistics, including frame count, object count, etc.
- **fused_4d.npz**: Contains point cloud, trajectory, and temporal information for 4D scene data
- **processing_log_*.log**: Detailed processing log for debugging

## Configuration

### Environment Variables
- `DASHSCOPE_API_KEY`: For calling Qwen API service
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices to use

### Main Parameters
- `--flow_model`: Optical flow model selection (unimatch/raft/gmflow)
- `--grid_size`: CoTracker grid size (default 50)
- `--interval`: Tracking interval frames (default 10)

## Notes

1. **GPU Requirements**: At least 8 A100 GPUs recommended for large model inference
2. **Storage Space**: Pre-trained models require approximately 100GB storage
3. **Memory Requirements**: Sa2VA-8B requires at least 32GB VRAM, Qwen2.5-VL requires more resources
4. **Data Formats**: Supports common video formats and image sequences

## Related Links

- [Sa2VA Project](https://lxtgh.github.io/project/sa2va)
- [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)

## License

This project is built upon multiple open-source projects. Please refer to the license requirements of each submodule.

## Contributing

Issues and Pull Requests are welcome. Before submitting code, please ensure:
- Code follows project style guidelines
- Passes all test cases
- Updates relevant documentation

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{wen2025dynamicverse,
        title={DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling}, 
        author={Kairun Wen and Yuzhi Huang and Runyu Chen and Hui Zheng and Yunlong Lin and Panwang Pan and Chenxin Li and Wenyan Cong and Jian Zhang and Junbin Lu and Chenguo Lin and Dilin Wang and Zhicheng Yan and Hongyu Xu and Justin Theiss and Yue Huang and Xinghao Ding and Rakesh Ranjan and Zhiwen Fan},
        year={2025},
        eprint={2512.03000},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2512.03000}, 
    }
```
