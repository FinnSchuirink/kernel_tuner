import shutil
from experiments.convolution_milo.convolution_milo import tune
import os

DEVICE = "NVIDIA GeForce GTX 1650"
LANG = "CUDA"

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