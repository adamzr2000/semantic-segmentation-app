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
    libsleef-dev \
    gdb \
    procinfo \
    python3-pip \
    python3-opencv \
    git \
    wget \
    && apt-get clean

# Install dependencies 
RUN pip3 install --no-cache-dir \
      keras==2.15.0 \
      tensorflow==2.15.0 \
      numpy \
      huggingface_hub \
      flask

RUN pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu


# Create a working directory
WORKDIR /app

COPY app/ /app/
