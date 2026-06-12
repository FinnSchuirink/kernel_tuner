#!/usr/bin/env python

###########################################################################################################
#                                                                                                         #
#                     This file belongs to Ben van Werkhoven, all rights are reserved                     #
#          Link: https://github.com/benvanwerkhoven/benchmark_kernels/blob/finn/pnpoly/pnpoly.cu          #
#                                                                                                         #
###########################################################################################################

""" Point-in-Polygon host/device code tuner

This program is used for auto-tuning the host and device code of a CUDA program
for computing the point-in-polygon problem for very large datasets and large
polygons.

The time measurements used as a basis for tuning include the time spent on
data transfers between host and device memory. The host code uses device mapped
host memory to overlap communication between host and device with kernel
execution on the GPU. Because each input is read only once and each output
is written only once, this implementation almost fully overlaps all
communication and the kernel execution time dominates the total execution time.

The code has the option to precompute all polygon line slopes on the CPU and
reuse those results on the GPU, instead of recomputing them on the GPU all
the time. The time spent on precomputing these values on the CPU is also
taken into account by the time measurement in the code.

This code was written for use with the Kernel Tuner. See:
     https://github.com/benvanwerkhoven/kernel_tuner

Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
"""
import numpy as np
import kernel_tuner
import os


def tune(compilation_cache_enabled=False):

    # set the number of points and the number of vertices
    size = np.int32(1e5)
    problem_size = (size, 1)
    vertices = 100

    # generate input data
    points = np.random.randn(2*size).astype(np.float32)
    bitmap = np.zeros(size).astype(np.int32)

    # as test input we use a circle with radius 1 as polygon and
    # a large set of normally distributed points around 0,0
    vertex_seeds = np.sort(np.random.rand(vertices)*2.0*np.pi)[::-1]
    vertex_x = np.cos(vertex_seeds)
    vertex_y = np.sin(vertex_seeds)
    vertex_xy = np.zeros(2*vertices, dtype=np.float32)
    np.copyto(vertex_xy, np.array( list(zip(vertex_x, vertex_y)) ).astype(np.float32).ravel())

    # setup constant memory
    c_mem = {}
    c_mem["d_vertices"] = vertex_xy

    # kernel arguments
    args = [bitmap, points, size]

    # setup tunable parameters
    tune_params = {}
    tune_params["block_size_x"]          = [32 * i for i in range(1, 9)]  # 8 values
    tune_params["tile_size"]             = [1, 2]                          # 2 values
    tune_params["between_method"]        = [0, 1]                          # 2 values
    tune_params["use_method"]            = [0, 1]                          # 2 values
    tune_params["loop_unroll_factor_v"] = [0] #+ [i for i in range(1, vertices+1) if vertices % i == 0]

    # tell Kernel Tuner how to compute the grid dimensions from the problem_size
    grid_div_x = ["block_size_x", "tile_size"]

    # start tuning
    cwd = os.path.dirname(os.path.realpath(__file__))
    kernel_file = os.path.join(cwd, "pnpoly_small.cu")
    results, env = kernel_tuner.tune_kernel("cn_pnpoly", kernel_file, problem_size, args, tune_params, grid_div_x=grid_div_x, cmem_args=c_mem,
                    verbose=True, strategy="random_sample", compilation_cache_enabled=compilation_cache_enabled, iterations=3)

    return results, env


if __name__ == "__main__":
    tune()