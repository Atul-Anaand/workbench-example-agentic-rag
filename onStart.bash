#!/bin/bash
# Runs on every container start (invoked by Workbench's entrypoint.sh).
# Purpose: self-heal runtime deps that may be missing if the container
# is recreated from an image that doesn't have them baked in.
#
# A no-op when everything is already installed (~1 second check).

set -e

MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# faster-whisper + CUDA libs for local audio transcription
if ! python3 -c "import faster_whisper" 2>/dev/null; then
    echo "[onStart] faster-whisper missing — installing..."
    pip install --user $MIRROR faster-whisper==1.2.1 nvidia-cublas-cu12
else
    echo "[onStart] faster-whisper already installed"
fi

# cublas shared lib (needed by ctranslate2 for GPU)
if [ ! -f "$HOME/.local/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12" ]; then
    echo "[onStart] cublas lib missing — installing..."
    pip install --user $MIRROR nvidia-cublas-cu12
else
    echo "[onStart] cublas lib already present"
fi

echo "[onStart] environment ready"
