#!/bin/bash

export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

echo "LOCAL_UID: $LOCAL_UID"
echo "LOCAL_GID: $LOCAL_GID"

# Clone ct_laboratory if it doesn't exist
if [ ! -d "ct_laboratory" ]; then
    echo "Cloning ct_laboratory..."
    git clone https://github.com/tivnanmatt/ct_laboratory.git
fi

# Clone gmi if it doesn't exist
if [ ! -d "gmi" ]; then
    echo "Cloning gmi..."
    git clone https://github.com/Generative-Medical-Imaging-Lab/gmi.git
fi

# Build the container
echo "Building container..."
docker compose build

# Start the container
echo "Starting container..."
docker compose up -d

# Build ct_laboratory inside the container (as non-root user)
echo "Building ct_laboratory..."
docker exec -it sparse_view_head_ct_container bash -c "cd /workspace/ct_laboratory && TORCH_CUDA_ARCH_LIST='8.0' CUDA_HOME=/usr/local/cuda make clean && TORCH_CUDA_ARCH_LIST='8.0' CUDA_HOME=/usr/local/cuda make"

# Install gmi inside the container (as non-root user with --user flag)
echo "Installing gmi..."
docker exec -it sparse_view_head_ct_container bash -c "cd /workspace/gmi && pip install --user -r requirements.txt && pip install --user -e ."

echo "Setup complete!"


