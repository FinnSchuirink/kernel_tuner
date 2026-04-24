import shutil
from convolution_milo import tune
import os

DEVICE = "CUDA"
LANG = "A100"

def _clear_cache(cache_dir="compilation_cache"):
    if (os.path.exists(cache_dir)):
        shutil.rmtree(cache_dir)

def _print_statistics(lang, cold_cache_results, warm_cache_results):
    warm_compile_time = warm_cache_results['compile_time']
    cold_compile_time = cold_cache_results['compile_time']

    print(f"{lang}\n")
    print(f"Total compilation time tune_kernel with a cold cache: {cold_compile_time}\n")
    print(f"Total compilation time tune_kernel with a warm cache: {warm_compile_time}\n")
    
    if (warm_compile_time == 0):
        print("Speedup: NaN, dividing by 0")
    else:
        speedup = cold_compile_time / warm_compile_time
        print (f"Speedup: {speedup}")

def _run_kernel_pipeline_twice():
    _clear_cache()
    cold, _ = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True)
    warm, _ = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True)
    return cold[0], warm[0]

def main():
    cold, warm = _run_kernel_pipeline_twice()
    _print_statistics(lang=LANG, cold_cache_results=cold, warm_cache_results=warm)

main()