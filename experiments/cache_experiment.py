import shutil
from convolution_milo import tune
import os

DEVICE = "NVIDIA GeForce GTX 1650"
LANG = "CUDA"

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
    print(f"    - Cold cache time: {sum_cold}s")
    print(f"    - Warm caache time: {sum_warm}s")
    
    if (sum_warm == 0):
        print("    - Speedup: NaN, warm cache time == 0")
    else:
        speedup = sum_cold / sum_warm
        print (f"    - Speedup: {speedup}x")

def _run_kernel_pipeline_twice():
    _clear_cache()
    cold, _ = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True)
    warm, _ = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True)
    return cold, warm

def main():
    cold, warm = _run_kernel_pipeline_twice()
    _print_statistics(lang=LANG, cold_cache_results=cold, warm_cache_results=warm)

main()