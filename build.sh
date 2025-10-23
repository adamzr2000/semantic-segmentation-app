#!/bin/bash

# Default values for CPU build
DOCKERFILE="Dockerfile"
IMAGE_TAG="semantic-segmentation-app"

# If --gpu flag is passed, switch to GPU Dockerfile and tag
if [[ "$1" == "--gpu" ]]; then
    echo "Building GPU Docker image..."
    DOCKERFILE="Dockerfile-gpu"
    IMAGE_TAG="semantic-segmentation-app:gpu"
else
    echo "Building CPU Docker image..."
    IMAGE_TAG="semantic-segmentation-app"
fi

# Build the Docker image
sudo docker build -f "$DOCKERFILE" . -t "$IMAGE_TAG"

