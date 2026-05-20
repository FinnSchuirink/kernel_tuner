import shutil
from experiments.convolution_milo.convolution_milo import tune
import os

DEVICE = "GTX1650"
LANG = "CUDA"

def _print_statistics(lang, cache_results):
    ## Calculate total compilation times
    sum = 0
    
    for result in cache_results:
        sum_cold += result.get('compile_time', 0)

    print(f"\n{lang} Results:")
    print(f"    - Cold cache time: {sum}s")
    
    ## FIXME: Compare with non cached modified runtime parameters

def _clear_cache(cache_dir="compilation_cache"):
    if (os.path.exists(cache_dir)):
        shutil.rmtree(cache_dir)

def _run_kernel_pipeline():
    _clear_cache()
    cache_results, _ = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True)
    return cache_results

def main():
    cache_results = _run_kernel_pipeline()

main()