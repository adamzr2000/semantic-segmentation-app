#!/bin/bash
#entrypoint="python3 test.py"

entrypoint="python3 main.py"

docker run \
    -it \
    --rm \
    --name semantic-segmentation-app \
    --hostname semantic-segmentation-app \
    -v "$(pwd)"/app:/app \
    --privileged \
    -p 5000:5000 \
    --group-add video \
    semantic-segmentation-app:latest \
    $entrypoint
