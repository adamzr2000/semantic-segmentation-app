#!/bin/bash

docker run \
    -it \
    --rm \
    --name semantic-segmentation-app \
    --hostname semantic-segmentation-app \
    -v "$(pwd)"/app:/app \
    --privileged \
    -p 5000:5000 \
    --group-add video \
    semantic-segmentation-app 

echo "Done."
