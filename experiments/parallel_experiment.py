#!/usr/bin/env python
import itertools
import time
from collections import OrderedDict

import numpy

from kernel_tuner import core
from kernel_tuner.interface import Options
from kernel_tuner.runners.parallel import ParallelRunner
from kernel_tuner.runners.sequential import SequentialRunner

def build(kernel_name, kernel_string, problem_size, args, tune_params, cmem_args):
    kernel_options = Options(
        kernel_name=kernel_name,
        kernel_string=kernel_string,
        problem_size=problem_size,
        arguments=args,
        lang=None,
        grid_div_x=["block_size_x", "tile_size_x"],
        grid_div_y=["block_size_y", "tile_size_y"],
        grid_div_z=None,
        smem_args=None,
        cmem_args=cmem_args,
        texmem_args=None,
        block_size_names=None,
        defines=None,
    )

    tuning_options = Options(
        tune_params=tune_params,
        restrictions=["use_padding==0 or (block_size_x % 32 != 0)"],
        answer=None,
        atol=1e-6,
        verify=None,
        verbose=False,
        objective="time",
        objective_higher_is_better=False,
        cache=None,
        metrics=None,
    )

    device_options = Options(
        device=0,
        platform=0,
        quiet=False,
        compiler=None,
        compiler_options=None,
    )

    return kernel_options, tuning_options, device_options

def main():

    ## Read kernel string
    with open('experiments/convolution_milo.cu', 'r') as f:
        kernel_string = f.read()
    kernel_name = "convolution_kernel"
    problem_size = (4096, 4096)

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["filter_height"] = [1, 3, 5]
    tune_params["filter_width"] = [1, 3, 5]
    tune_params["block_size_x"] = [16, 32]
    tune_params["block_size_y"] = [8, 16]
    tune_params["tile_size_x"] = [1]
    tune_params["tile_size_y"] = [1]

    tune_params["use_padding"] = [0,1]  #toggle the insertion of padding in shared memory
    tune_params["read_only"] = [0,1]    #toggle using the read-only cache

    largest_fh = max(tune_params["filter_height"])
    largest_fw = max(tune_params["filter_width"])

    input_size = ((problem_size[0]+largest_fw-1) * (problem_size[1]+largest_fh-1))
    image_size = numpy.prod(problem_size)

    output_image = numpy.zeros(image_size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    cmem_args = {'d_filter': filter_weights}
    args = [output_image, input_image, filter_weights]

    # create KernelSource
    kernel_source = core.KernelSource(kernel_name,
                                    kernel_string,
                                    lang=None,
                                    defines=None)

    # Create the searchspace through cartesian product of tune parameters
    searchspace = list(
        itertools.product(
            tune_params["filter_height"],
            tune_params["filter_width"],
            tune_params["block_size_x"],
            tune_params["block_size_y"],
            tune_params["tile_size_x"],
            tune_params["tile_size_y"],
            tune_params["use_padding"],
            tune_params["read_only"],
        )
    )

    ## SEQUENTIAL
    kernel_options, tuning_options, device_options = build(kernel_name, kernel_string, problem_size, args, tune_params, cmem_args)

    seq_runner = SequentialRunner(kernel_source, kernel_options, device_options, 1, None)
    seq_runner.warmed_up = False

    # Measure time sequential runner
    seq_start = time.perf_counter()
    seq_results = seq_runner.run(searchspace, tuning_options)
    seq_wall = time.perf_counter() - seq_start

    ## PARALLEL
    kernel_options, tuning_options, device_options = build(kernel_name, kernel_string, problem_size, args, tune_params, cmem_args)

    parallel_runner = ParallelRunner(kernel_source, kernel_options, device_options, 1, None)
    parallel_runner.warmed_up = False

    ## Measure time parallel runner
    parallel_start = time.perf_counter()
    parallel_results = parallel_runner.run(searchspace, tuning_options)
    parallel_wall = time.perf_counter() - parallel_start

    print("\nSequential wall time: ", round(seq_wall, 3), "s")
    print("\nParallel wall time: ", round(parallel_wall, 3), "s")

## Only execute directly
if __name__ == "__main__":
    main()