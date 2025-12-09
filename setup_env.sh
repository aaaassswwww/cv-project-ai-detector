#!/bin/bash

# AI图像鉴别项目 - 环境自动配置脚本
# 使用方法: bash setup_env.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "AI图像鉴别项目 - 环境配置脚本"
echo "=========================================="
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未检测到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

echo "✓ 检测到 conda"
echo ""

# 检查环境是否已存在
if conda env list | grep -q "^cvpj "; then
    echo "⚠️  警告: 环境 'cvpj' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除旧环境..."
        conda deactivate 2>/dev/null || true
        conda env remove -n cvpj -y
        echo "✓ 旧环境已删除"
    else
        echo "跳过环境创建"
        exit 0
    fi
fi

echo "=========================================="
echo "步骤 1/3: 创建 Conda 环境"
echo "=========================================="
echo "环境名称: cvpj"
echo "Python 版本: 3.12"
echo ""

# 创建环境
conda create -n cvpj python=3.12 -y

echo ""
echo "✓ Conda 环境创建成功"
echo ""

echo "=========================================="
echo "步骤 2/3: 激活环境并安装 PyTorch"
echo "=========================================="
echo ""

# 激活环境并安装依赖
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cvpj

# 检测是否有GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA GPU，安装 CUDA 版本的 PyTorch..."
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "⚠️  未检测到 GPU，安装 CPU 版本的 PyTorch..."
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

echo ""
echo "✓ PyTorch 安装成功"
echo ""

echo "=========================================="
echo "步骤 3/3: 安装其他依赖包"
echo "=========================================="
echo ""

pip install -r requirements.txt

echo ""
echo "✓ 所有依赖安装完成"
echo ""

echo "=========================================="
echo "验证安装"
echo "=========================================="
echo ""

python -c "
import torch
import torchvision
import numpy as np
import matplotlib
import sklearn
from PIL import Image
from tqdm import tqdm

print('✓ PyTorch 版本:', torch.__version__)
print('✓ torchvision 版本:', torchvision.__version__)
print('✓ CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA 版本:', torch.version.cuda)
    print('✓ GPU 设备:', torch.cuda.get_device_name(0))
print('✓ NumPy 版本:', np.__version__)
print('✓ Matplotlib 版本:', matplotlib.__version__)
print('✓ scikit-learn 版本:', sklearn.__version__)
print('')
print('🎉 所有依赖包验证通过！')
"

echo ""
echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "接下来的步骤："
echo "1. 激活环境: conda activate cvpj"
echo "2. 准备数据集（参见 SETUP_GUIDE.md）"
echo "3. 开始训练: python src/train.py --output_dir checkpoints"
echo ""
echo "详细使用说明请查看 SETUP_GUIDE.md"
echo ""
