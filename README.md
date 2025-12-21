
<h2 align="center"> <a href="https://arxiv.org/abs/2512.03000">DynamicVerse: A Physically-Aware Multimodal Framework<br>for 4D World Modeling</a></h2>

<h5 align="center">

[![Home Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://dynamic-verse.github.io/) [![arXiv](https://img.shields.io/badge/Arxiv-251203.03000-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2512.03000) [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00)](https://huggingface.co/datasets/kairunwen/DynamicVerse) [![youtube](https://img.shields.io/badge/Demo_Video-E33122?logo=Youtube)](https://www.youtube.com/watch?v=0h7XysIpG8Y) [![X](https://img.shields.io/badge/-Twitter@Kairun%20Wen%20-black?logo=twitter&logoColor=1D9BF0)](https://x.com/KairunWen) 
</h5>

<br>

https://github.com/user-attachments/assets/61c818e1-9258-4bb1-9f9d-c656dfd5ce8a

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Run DynamicGen Demo](#run-dynamicgen-demo)
  - [Qwen2.5-VL Configuration](#qwen25-vl-configuration)
- [Processing Pipeline](#processing-pipeline)
- [Output Directory Structure](#output-directory-structure)
  - [Output Files Description](#output-files-description)
- [Evaluation](#evaluation)
  - [Preprocessing](#preprocessing)
  - [Metrics Evaluation](#metrics-evaluation)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)

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
├── dynamicBA/              # 4D scene reconstruction module
│   ├── unimatch/           # Optical flow and depth estimation
│   ├── dataset_prepare/    # Data preprocessing tools
│   └── config/             # Configuration files
├── data/                   # Dataset directory
├── scripts/                # Preprocessing scripts
├── dynamicgen/             # Pipeline execution
│   └── scripts/            # DynamicGen pipeline
├── Sa2VA/                  # Vision-language multimodal model
├── CoTracker/              # Point tracking model
├── UniDepth/               # Monocular depth estimation
└── ...
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

# Specify model name
model="Qwen/Qwen2.5-VL-72B-Instruct"
```

**Install Dependencies:**
```bash
pip install accelerate
pip install qwen-vl-utils==0.0.14
uv pip install -U vllm  # Requires vllm>=0.11.0
```

**Start Local Service:**
```python
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

For detailed deployment instructions, refer to [Qwen-VL](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file)


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
        │   ├── pose.npz        # Intrinsic and Extrinsics
        │   ├── depth/          # Depth maps
        │   └── flow/           # Optical flow data
        └── processing_log_<scene_id>.log  # Processing log
```

### Output Files Description

- **dynamic_objects_*.json**: Contains detected dynamic object information, including position, category, and tracking ID
- **instance_labels.json**: Label mapping for each instance, used for multi-object segmentation
- **result_summary.json**: Segmentation result statistics, including frame count, object count, etc.
- **processing_log_*.log**: Detailed processing log for debugging

## Evaluation
We provide preprocessed datasets to reproduce  Table 1 and 2 in our main paper.

### Preprocessing
You can also download our preprocessed data that we used for the quantitive results in our paper:
```bash
cd data
gdown https://drive.google.com/uc?id=1V1WIRvnJCJStL63rluwNZMPI2Gq4-yQy -O preprocessed.zip
unzip preprocessed.zip
```

### Metrics Evaluation
We provide evaluation scripts for pose and depth metrics:
```bash
bash ./scripts/eval.sh
```

## Notes

1. **Storage Space**: Pre-trained models require approximately 100GB storage
2. **Memory Requirements**: Sa2VA-8B requires at least 32GB VRAM, Qwen2.5-VL requires more resources
3. **Data Formats**: Supports common video formats and image sequences

## Acknowledgements
Our code is based on the following awesome repositories:
- [Sa2VA](https://lxtgh.github.io/project/sa2va)
- [Qwen-VL](https://github.com/QwenLM/Qwen3-VL)
- [Cotracker](https://github.com/Davidyao99/co-tracker/tree/9e3082b6c921fa9546535d0190f9693fab193c31)
- [Unidepth](https://github.com/lpiccinelli-eth/UniDepth)
- [Uni4D](https://github.com/Davidyao99/uni4d)
- [DAM](https://github.com/NVlabs/describe-anything)
- [CameraBench](https://github.com/sy77777en/CameraBench)

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
