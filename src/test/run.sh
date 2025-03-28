#!/usr/bin/env bash

lib='/home/paul/ETH/HS22/DPHPC/gpuless/src/cmake-build-debug/libgpuless.so'

basic_targets=(
    unit/simple
    unit/objkernel
    unit/manykernel
    unit/manymodule
)

exec_type=tcp
log_level=debug
unit_only=no

if [[ $2 == "debug" ]]; then
    log_level=debug
fi

if [[ $2 == "tcp" ]]; then
    exec_type=tcp
fi

if [[ $3 == "unit" ]]; then
    unit_only=yes
fi


run_unit() {
    cuda_bin="$1"
    printf "cuda_bin: ${cuda_bin}\n"
    target="$2"
    printf "running: ${target}\n\n"
    SPDLOG_LEVEL=${log_level} EXECUTOR_TYPE=${exec_type} CUDA_BINARY=./${cuda_bin} LD_PRELOAD=${lib} ./${target}
    printf "\n"
}

run_custom() {
    cuda_bin="$1"
    target="$2"
    dir="$3"
    printf "running custom: ${dir}\n"
    pushd . >/dev/null
    cd "$dir"
    run_unit "$cuda_bin" "$target"
    popd >/dev/null
}

# unit tests
for target in ${basic_targets[@]}; do
    run_unit "$target" "$target"
done

