#!/usr/bin/bash

LD_PRELOAD="$HOME/gpuless/src/build/libgpuless.so" \
	CUDA_VISIBLE_DEVICES=0 \
EXECUTOR_TYPE=tcp \
  MANAGER_IP=127.0.0.1 \
  MANAGER_PORT=8002 \
  CUDA_BINARY=$HOME/libtorch/lib/libtorch_cuda.so \
  python run-batched.py