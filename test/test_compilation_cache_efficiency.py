import numpy
from kernel_tuner import tune_kernel
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)

# Kernels ----------------------------------------------------------------------------------

C_KERNEL = """ 
extern "C" float complex_transform(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        float x = a[i];
        float y = b[i];
        float v = x + y;
        v = v * v + 0.5f * x - 0.25f * y;
        v = v * 1.0001f + x * y;
        v = v + (x - y) * (x + y);
        v = v * 0.75f + x * 0.125f + y * 0.0625f;
        out[i] = v;
    }
    return 0.0f;
}
"""

CUDA_KERNEL = """
__global__ void complex_transform(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float y = b[i];
        float v = x + y;
        v = v * v + 0.5f * x - 0.25f * y;
        v = v * 1.0001f + x * y;
        v = v + (x - y) * (x + y);
        v = v * 0.75f + x * 0.125f + y * 0.0625f;
        out[i] = v;
    }
}
"""

FORTRAN_KERNEL = """
real(c_float) function complex_transform(out, a, b, n) bind(C, name="complex_transform_")
    use iso_c_binding
    implicit none
    integer(c_int), value :: n
    real(c_float), intent(out) :: out(n)
    real(c_float), intent(in)  :: a(n), b(n)
    integer :: i
    real(c_float) :: x, y, v
    do i = 1, n
        x = a(i)
        y = b(i)
        v = x + y
        v = v * v + 0.5 * x - 0.25 * y
        v = v * 1.0001 + x * y
        v = v + (x - y) * (x + y)
        v = v * 0.75 + x * 0.125 + y * 0.0625
        out(i) = v
    end do
    complex_transform = 0.0
end function complex_transform
"""

# Constants --------------------------------------------------------------------------------

TUNE_PARAMS = {"block_size_x": [1, 2, 4, 8, 16]}

ITERATIONS = 1

# Helper functions -------------------------------------------------------------------------


def _make_args(size=50000):

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    return c, a, b, n

def _clear_cache(cache_dir="compilation_cache"):
    if (os.path.exists(cache_dir)):
        shutil.rmtree(cache_dir)

def _print_statistics(lang, cold_cache_results, warm_cache_results):

    ## Calculate total compilation times
    sum_cold = 0
    sum_warm = 0

    for result in warm_cache_results:
        sum_warm += result.get('compile_time', 0)
    
    for result in cold_cache_results:
        sum_cold += result.get('compile_time', 0)

    print(f"\n{lang} Results:")
    print(f"    - Cold cache time: {sum_cold}ms")
    print(f"    - Warm cache time: {sum_warm}ms")
    
    if (sum_warm == 0):
        print("    - Speedup: NaN, warm cache time == 0")
    else:
        speedup = sum_cold / sum_warm
        print (f"    - Speedup: {speedup}x")

def _run_kernel_pipeline_twice(tune_kwargs):
    logging.info("\nRunning cold cache build:")
    cold, _ = tune_kernel(**tune_kwargs)

    logging.info("\nRunning warm cache build:")
    warm, _ = tune_kernel(**tune_kwargs)
    return cold, warm

# C ----------------------------------------------------------------------------------------

def _test_c_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=C_KERNEL,
        problem_size=n,
        arguments=[c, a, b, n],
        tune_params=TUNE_PARAMS, 
        lang="C",
        compiler="g++",
        compiler_options=["-O2"],
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="C", cold_cache_results=cold, warm_cache_results=warm)

# FORTRAN ----------------------------------------------------------------------------------

def _test_fortran_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=FORTRAN_KERNEL,
        problem_size=n,
        arguments=[c, a, b, n],
        tune_params=TUNE_PARAMS,
        lang="FORTRAN",
        compiler="gfortran",
        compiler_options=["-O2"],
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="FORTRAN", cold_cache_results=cold, warm_cache_results=warm)

# CUDA -------------------------------------------------------------------------------------

def _test_cuda_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=CUDA_KERNEL,
        problem_size=n,
        arguments=[c, a, b, n],
        tune_params=TUNE_PARAMS,
        lang="CUDA",
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="CUDA", cold_cache_results=cold, warm_cache_results=warm)

# CUPY -------------------------------------------------------------------------------------

def _test_cupy_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=CUDA_KERNEL,
        problem_size=n,
        arguments =[c, a, b, n],
        tune_params=TUNE_PARAMS,
        lang="CUPY",
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="CUPY", cold_cache_results=cold, warm_cache_results=warm)

# NVCUDA -----------------------------------------------------------------------------------

def _test_nvcuda_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=CUDA_KERNEL,
        problem_size=n,
        arguments =[c, a, b, n],
        tune_params=TUNE_PARAMS,
        lang="NVCUDA",
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="NVCUDA", cold_cache_results=cold, warm_cache_results=warm)

# HIP --------------------------------------------------------------------------------------

def _test_hip_caching():
    c, a, b, n = _make_args()
    kwargs = dict(
        kernel_name="complex_transform",
        kernel_source=CUDA_KERNEL,
        problem_size=n,
        arguments =[c, a, b, n],
        tune_params=TUNE_PARAMS,
        lang="HIP",
        iterations=ITERATIONS,
        answer=[None, None, None, None],
        verbose=True,
    )
    cold, warm = _run_kernel_pipeline_twice(kwargs)
    _print_statistics(lang="HIP", cold_cache_results=cold, warm_cache_results=warm)

def main():
    tests = {
        "c": _test_c_caching,
        "fortran": _test_fortran_caching,
        "cuda": _test_cuda_caching,
        "cupy": _test_cupy_caching,
        "nvcuda": _test_nvcuda_caching,
        #"hip": _test_hip_caching,
    }
    print("\n" + "=" * 70)
    print("Compilation cache efficiency test")
    print("=" * 70)
    _clear_cache()
    for test in tests:
        print(f"\n{'-' * 70}")
        print(f"Testing: {test.upper()}")
        print(f"{'-' * 70}")
        try:
            tests[test]()
        except Exception as e:
            print(f"Error when testing caching on {test.upper()}: {e}")

main()