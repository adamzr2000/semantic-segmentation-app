#!/usr/bin/env bash
find . -mindepth 1 -maxdepth 1 ! -name '.gitkeep' ! -name 'clean.sh' -exec rm -rf -- {} +

