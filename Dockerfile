FROM ubuntu:22.04

LABEL maintainer="azahir@pa.uc3m.es"

# Set bash as default shell, and avoid prompts during package installation
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    libgomp1 \
    libsleef3 \
    libtbb2 \    
    gdb \
    procinfo \
    python3-pip \
    python3-opencv \
    git \
    wget \
    && apt-get clean


# Upgrade pip first
RUN python3 -m pip install --upgrade pip

# Install torch/torchvision
RUN pip3 install \
	torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install \
      numpy==1.26.4 \
      pillow==10.4.0 \
      opencv-python-headless==4.10.0.84 \
      flask==3.0.3

# Create a working directory
WORKDIR /app

# Copy your app
COPY app/ .
