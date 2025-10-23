#!/bin/bash
#entrypoint="python3 test.py"

#entrypoint="python3 main.py"

entrypoint="/bin/bash"

docker run \
  -it \
  --rm \
  --name semantic-segmentation-app \
  --hostname semantic-segmentation-app \
  -v "$(pwd)"/app:/app \
  -e MODEL=sol-cpu \
  -e LD_LIBRARY_PATH=/app/models/sol_deeplabv3_resnet50/lib_cpu:${LD_LIBRARY_PATH} \
  --privileged \
  -p 5000:5000 \
  --group-add video \
  semantic-segmentation-app:latest \
  $entrypoint
