#!/bin/bash
# This file contains bash commands that will be executed at the beginning of the container build process.
# Note: This file may be removed if you don't need to use it

pip install --upgrade pip

# Write pip config so pip installs use the Tsinghua mirror.
mkdir -p "$HOME/.config/pip"
cat > "$HOME/.config/pip/pip.conf" << 'PIPCONF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
retries = 15
resume-retries = 30
PIPCONF

# ENTERPRISE OPTIMIZATION: Force CPU-only PyTorch 
# This prevents pip from downloading 3.5GB of unnecessary nvidia-cuda packages
# because inference is handled by Ollama on the host.
pip install --user torch --index-url https://download.pytorch.org/whl/cpu
