import shutil
from experiments.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "GTX1650"
LANG = "CUDA"
NUM_ITERATIONS = 5
RESULTS_LOC = f"results/parallel_results_{DEVICE}_{LANG}.json"
PYCACHE = "__pycache__"

def _remove_base_cache():
    base = f"cachefiles/{DEVICE}"
    suffix = [".json", "-results.json", "-metadata.json"]
    for s in suffix:
        file = base + s
        if (os.path.exists(file)):
            os.remove(file)
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)

def _single_tune(runner_mode: bool):
    start = time.perf_counter()
    results, env = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True, runner_mode=runner_mode)
    wall = time.perf_counter() - start
    return results, env, wall

def _extract_stats(results, env, wall_time):
    total_compile = env["total_compile_time"] / 1000.0
    total_benchmark = env["total_benchmark_time"] / 1000.0
    total_framework = env["total_framework_time"] / 1000.0
    total_strategy = env["total_strategy_time"] / 1000.0
    total_overhead = env["overhead_time"] / 1000.0

    cache_stats = env["compilation_cache_stats"]
    cache_hits = cache_stats.get("hits")

    individual_compile_times = []
    for r in results:
        individual_compile_times.append(r.get("compile_time"))
    
    return {
        "total_compile": total_compile,
        "total_benchmark": total_benchmark,
        "total_framework": total_framework,
        "total_strategy": total_strategy,
        "total_overhead": total_overhead,

        "total_wallclock": wall_time,
        "compile_fraction": total_compile / wall_time if wall_time > 0 else float("NaN"),

        "individual_compile_times": individual_compile_times,

        "n_configs": len(results),
        "cache_hits": cache_hits,
    }

def _run_N_times(stats: list[dict]):
    keys = [
        "total_compile",
        "total_benchmark",
        "total_framework",
        "total_strategy",
        "total_overhead",
        "total_wallclock",
        "compile_fraction",
    ]
    aggregate = {}

    for key in keys:
        values = []
        for stat in stats:
            values.append(stat[key])
        aggregate[f"{key}_mean"] = np.mean(values)
        aggregate[f"{key}_std.dev"] = np.std(values, ddof = 1)

    return aggregate

def _calculate_speedup(seq_mean, parallel_mean):
    if seq_mean == 0:
        return float("NaN"), float("NaN")
    
    speedup = seq_mean / parallel_mean
    return speedup


def _save_results(seq: list, parallel: list):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "device": DEVICE,
                "language": LANG,
                "cold": seq,
                "warm": parallel,
                "speedup": _calculate_speedup(seq["total_compile_mean"], parallel["total_compile_mean"])
            }, 
            f,
            indent=2
        )


def main():
    sequential_stats, parallel_stats = [], []

    for i in range(NUM_ITERATIONS):
        _remove_base_cache()
        
        ## SEQUENTIAL
        results, env, wall = _single_tune(runner_mode="Sequential")
        sequential_stats.append(_extract_stats(results, env, wall))

        ## PARALLEL
        results, env, wall = _single_tune(runner_mode="Parallel")
        parallel_stats.append(_extract_stats(results, env, wall))

    sequential_aggregate = _run_N_times(sequential_stats)
    parallel_aggregate = _run_N_times(parallel_stats)

    _save_results(sequential_aggregate, parallel_aggregate)


main()