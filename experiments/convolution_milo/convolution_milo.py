#!/usr/bin/env python

##################################################################################
#                                                                                #
#       This file belongs to Floris-Jan Willemsen, all rights are reserved       #
#                 Link: https://github.com/AutoTuningAssociation                 #
#                                                                                #
##################################################################################

import os
from collections import OrderedDict
import time

import numpy

import kernel_tuner
from kernel_tuner.file_utils import store_metadata_file, store_output_file

def verify_results(parallel_results: list, seq_results: list, tune_params: list):
    """Check if both runner produce the same results to verify correctness.

        :param parallel_results: The results of the ParallelRunner
        :type parallel_results: list

        :param seq_results: The results of the SequentialRunner
        :type seq_results: list

        :param tune_params: List of all parameters that were tuned in the runs
        :type tune_params: list
    """
    param_keys = list(tune_params.keys())

    seq_configs = len(seq_results)
    parallel_configs = len(parallel_results)

    print("\nParallel configurations: ", seq_configs)
    print("\nSequential configurations: ", parallel_configs)

    if (seq_configs != parallel_configs):
        print("Differing amount of configurations evaluated!")
        return
    
    ## Get fastest configurations
    seq_best = min(seq_results, key=lambda r: r["time"])
    parallel_best = min(parallel_results, key=lambda r: r["time"])

    seq_params = {k: seq_best[k] for k in param_keys}
    parallel_params = {k: parallel_best[k] for k in param_keys}
    
    print(f"\nSequential best {seq_params} in {round(seq_best["time"], 3)}ms")
    print(f"\nParallel best {parallel_params} in {round(parallel_best["time"], 3)}ms")

    if (seq_params == parallel_params):
        print("Solutions found are equal!")
    elif (abs(seq_best["time"] - parallel_best["time"]) < 0.01):
        ## Sequential evaluates solutions in order of submission, while parallel in order of completion
        print("Different solutions were found, but are equally optimal")
    else:
        print("Solutions found are not equal!")


def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2) / 1e9


unit = "GFLOP"
w = h = 4096
fw = fh = 15
inputs = [w, h, fw, fh]
total_flops = ops(w, h, fw, fh)


# def tune(inputs, lang, strategy):
def tune(
    device_name: str,
    runner_mode: str,
    strategy="brute_force",
    strategy_options=None,
    verbose=True,
    quiet=False,
    lang="CUDA",
):
    if lang == "CUDA":
        kernel_file = "/convolution_milo.cu"
    elif lang == "HIP":
        kernel_file = "/convolution_milo.cu.hip"

    with open(os.path.dirname(os.path.realpath(__file__)) + kernel_file, "r") as f:
        kernel_string = f.read()

    # setup tunable parameters
    tune_params = OrderedDict()

    # tune_params["pwr_limit"] = get_pwr_limit(pwr_limit, 0)

    image_width, image_height, filter_width, filter_height = inputs

    tune_params["block_size_x"] = [16 * i for i in range(1, 2)]
    tune_params["block_size_y"] = [2**i for i in range(2)]
    tune_params["tile_size_x"] = [i for i in range(1, 2)]
    tune_params["tile_size_y"] = [i for i in range(1, 2)]
    tune_params["read_only"] = [0, 1]  # toggle using the read-only cache

    # do dry run
    # tune_params["nvml_gr_clock"] = [2100]
    # tune_params["block_size_x"] = [16]
    # tune_params["block_size_y"] = [1]
    # tune_params["tile_size_x"] = [1, 2, 4]
    # tune_params["tile_size_y"] = [1]
    # tune_params["read_only"] = [1]    #toggle using the read-only cache

    tune_params["use_padding"] = [0, 1]  # toggle the insertion of padding in shared memory

    # added based on cache and T1 file
    tune_params["use_shmem"] = [0, 1]
    tune_params["use_cmem"] = [1]
    tune_params["filter_height"] = [15]
    tune_params["filter_width"] = [15]
    restrict = [
        "use_padding==0 or block_size_x % 32 != 0",
        "block_size_x*block_size_y<=1024",
        "use_padding==0 or use_shmem != 0",
        "use_shmem == 0 or (((block_size_x*tile_size_x+(filter_width-1)))*((block_size_y*tile_size_y+(filter_height-1)))) < 12*1024",
    ]

    # # limit the search to only use padding when its effective
    # restrict = [
    #     "(use_padding==0 or (block_size_x % 32 != 0))",
    #     "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)",
    # ]
    # restrict.append(
    #     "(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)"
    #     % (filter_width - 1, filter_height - 1)
    # )
    # restrict.append("block_size_x * block_size_y <= 1024")
    # print(restrict)

    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = (problem_size[0] + largest_fw - 1) * (problem_size[1] + largest_fh - 1)

    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    cmem_args = {"d_filter": filter_weights}
    args = [output_image, input_image, filter_weights]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    total_flops = ops(*inputs)
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)

    base_cachepath = f"cachefiles/convolution_milo/{device_name.upper()}"

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "convolution_kernel",
        kernel_string,
        problem_size,
        args,
        tune_params,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        cmem_args=cmem_args,
        restrictions=restrict,
        cache=base_cachepath,
        metrics=metrics,
        lang=lang,
        iterations=32,
        device=0,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        runner_mode=runner_mode,
    )
    end = time.time()
    env["execution_time"] = end - start

    store_output_file(f"{base_cachepath}-{runner_mode}-results.json", results, tune_params)
    store_metadata_file(f"{base_cachepath}-{runner_mode}-metadata.json")
    return results, env

