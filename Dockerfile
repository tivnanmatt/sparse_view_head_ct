# FROM python:3.12-slim
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Install system dependencies using apt package manager
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git build-essential libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libxcb1 qtbase5-dev libqt5gui5 libqt5widgets5 libqt5core5a && rm -rf /var/lib/apt/lists/*

# Install CUDA development tools for compiling CUDA extensions
RUN apt-get update && apt-get install -y cuda-toolkit-12-8 && rm -rf /var/lib/apt/lists/*

# Make python3 and pip3 symlinks
RUN ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Upgrade pip python package manager
RUN pip install --upgrade pip

# Install PyTorch 12.8 with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ninja for faster builds
RUN pip install ninja

# Set PyTorch library path in environment
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Cache buster for user creation - rebuild from here if UID/GID changes
ARG CACHEBUST=1

# Create a non-root user with the same UID/GID as the host user
# If not specified, use 1000 as default user id and group id
# 
# For Linux hosts: Pass your UID/GID to ensure files created in the container are owned by your user
#    Example: docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t gmi-image .
#    Or use docker-compose: docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
#
# For macOS/Windows: Docker Desktop handles file permissions automatically, so you can use defaults
#    Example: docker build -t gmi-image .
#    Or use docker-compose: docker compose build

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -o -g $GROUP_ID -f user && \
    useradd -o -u $USER_ID -g $GROUP_ID -m -s /bin/bash user

# Create gmi_base directory and set ownership
RUN mkdir -p /workspace && chown user:user /workspace

# Set working directory
WORKDIR /workspace

# Note: ct_laboratory and gmi will be mounted from host workspace
# The repositories will be cloned and built on the host side

# Change ownership of workspace (will be updated when mounted)
RUN chown -R user:user /workspace/

# # Copy requirements.txt and install Python dependencies
# COPY requirements.txt /gmi_base/
# RUN pip install -r requirements.txt

# # Copy GMI source code
# COPY gmi/ /gmi_base/gmi/
# COPY setup.py /gmi_base/

# # Install GMI package in editable mode
# RUN pip install -e .

# # Switch to non-root user
USER user

# Set default command
CMD ["tail", "-f", "/dev/null"] 
