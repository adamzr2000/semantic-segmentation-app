#!/bin/bash

docker run \
    -it \
    --rm \
    --name semantic-segmentation-app \
    --hostname semantic-segmentation-app \
    -v "$(pwd)"/app:/app \
    --privileged \
    -p 5000:5000 \
    --runtime=nvidia \
    --group-add video \
    --gpus all \
    semantic-segmentation-app:gpu 

echo "Done."
