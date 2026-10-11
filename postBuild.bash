#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
sudo -E apt-get update && sudo -E apt-get install ffmpeg libsm6 libxext6 -y
pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple pytest
pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple gradio==4.44.1 fastapi==0.112.4 starlette==0.38.6 huggingface_hub==0.29.3
pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple faster-whisper==1.2.1 nvidia-cublas-cu12
