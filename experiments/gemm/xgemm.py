#!/usr/bin/env python
import os
import time
from collections import OrderedDict

import numpy
import kernel_tuner
from kernel_tuner.file_utils import store_output_file, store_metadata_file

def ops(m, n, k):
    return (2 * m * n * k) / 1e9 

M = N = K = 4096
alpha = numpy.float32(1.0)
beta  = numpy.float32(0.0)
total_flops = ops(M, N, K)


def tune(
    device_name: str,
    strategy="brute_force",
    strategy_options=None,
    verbose=True,
    quiet=False,
    simulation_mode=False,
    lang="CUDA"
):
    if lang == "CUDA":
        kernel_file = "/gemm.cu"
    
    with open(os.path.dirname(os.path.realpath(__file__)) + kernel_file, "r") as f:
        kernel_string = f.read()

    # setup tunable parameters
    tune_params = OrderedDict()

    # Compile time
    tune_params["TILE_M"]      = [2, 4, 8]
    tune_params["TILE_N"]      = [2, 4, 8]
    tune_params["TILE_K"]      = [8, 16] 
    tune_params["BLOCK_SIZE"]  = [16]

    # Runtime
    tune_params["nvml_gr_clock"]  = [1200, 1500, 1800, 2100]
    tune_params["nvml_mem_clock"] = [5000, 6000, 7000]

    restrict = []

    problem_size = (N, M)

    A = numpy.random.randn(M, K).astype(numpy.float32)
    B = numpy.random.randn(K, N).astype(numpy.float32)
    C = numpy.zeros((M, N),   dtype=numpy.float32)

    args = [A, B, C, numpy.int32(M), numpy.int32(N), numpy.int32(K), alpha, beta]

    grid_div_x = ["BLOCK_SIZE", "TILE_N"]
    grid_div_y = ["BLOCK_SIZE", "TILE_M"]

    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)

    base_cachepath = f"cachefiles/{device_name.upper()}"

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "gemm_kernel",
        kernel_string,
        problem_size,
        args,
        tune_params,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        restrictions=restrict,
        cache=base_cachepath,
        metrics=metrics,
        lang="CUDA",
        iterations=32,
        device=0,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
    )
    end = time.time()
    env["execution_time"] = end - start

    store_output_file(f"{base_cachepath}-results.json", results, tune_params)
    store_metadata_file(f"{base_cachepath}-metadata.json")

    return results, env


if __name__ == "__main__":
    import sys

    device = sys.argv[1]
    language = sys.argv[2]

    if len(sys.argv) != 3:
        raise ValueError(f"Usage: python gemm_cache_experiment.py [device_name] [language], given: {sys.argv}")
    
    if (language != "CUDA"):
        raise ValueError(f"{language} not valid, specify CUDA")

    tune(device_name=device, lang=language)
