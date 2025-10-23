#!/bin/bash
#entrypoint="python3 test.py"

entrypoint="python3 main.py"

docker run \
    -it \
    --rm \
    --name semantic-segmentation-app \
    --hostname semantic-segmentation-app \
    -v "$(pwd)"/app:/app \
    -e MODEL=sol-cpu \
    -e LD_LIBRARY_PATH=/app/models/sol_deeplabv3_resnet50/lib_cpu$LD_LIBRARY_PATH \
    -e OMP_NUM_THREADS=16 \
    -e MKL_NUM_THREADS=16 \
    -e OPENBLAS_NUM_THREADS=16 \
    -e NUMEXPR_NUM_THREADS=16 \
    -e BLIS_NUM_THREADS=16 \
    -e VECLIB_MAXIMUM_THREADS=16 \
    -e TBB_NUM_THREADS=16 \
    --privileged \
    -p 5000:5000 \
    --group-add video \
    semantic-segmentation-app:latest \
    $entrypoint
