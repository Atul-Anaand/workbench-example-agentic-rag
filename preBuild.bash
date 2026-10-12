#!/bin/bash
# This file contains bash commands that will be executed at the beginning of the container build process,
# before any system packages or programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it
pip install --upgrade pip

# Write pip.conf so all pip installs (including the one from requirements.txt)
# use the Tsinghua mirror. We do this in a config file rather than in
# requirements.txt because Workbench's wb-svc daemon strips the space
# after "--" options in requirements.txt, breaking "--index-url https://..."
mkdir -p "$HOME/.config/pip"
cat > "$HOME/.config/pip/pip.conf" << 'PIPCONF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
retries = 15
resume-retries = 30
PIPCONF
echo "pip.conf written to $HOME/.config/pip/pip.conf"
