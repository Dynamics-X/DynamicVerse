pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# # -------------------------- For qwen+sa2va -------------------------
pip install -r scripts/requirements.txt

# -------------------------- For Unidepth-------------------------
cd UniDepth
pip install . --no-deps
pip install einops
pip install wandb
pip install xformers==0.0.29
wget https://raw.githubusercontent.com/AbdBarho/xformers-wheels/refs/heads/main/xformers/components/attention/nystrom.py -O ./unidepth/layers/nystrom.py  # To ensure compatibility of pytorch 2.5.1
sed -i 's/from xformers\.components\.attention import NystromAttention/from .nystrom import NystromAttention/g' unidepth/layers/nystrom_attention.py
cd ..

conda install -c conda-forge ffmpeg libiconv
# -------------------------- For dynamicBA -------------------------

pip install imageio
pip install configargparse
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# -------------------------- For Evaluation -------------------------
pip install evo

