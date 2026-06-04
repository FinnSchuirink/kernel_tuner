import shutil
from experiments.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "A4000-Ada"
LANG = "CUDA"
NUM_ITERATIONS = 5
RESULTS_LOC = f"results/cache_experiment_between_runs_{DEVICE}_{LANG}.json"
PYCACHE = "__pycache__"
COMPILATION_CACHE_DIR = "compilation_cache"

def _clear_cache():
    if (os.path.exists(COMPILATION_CACHE_DIR)):
        shutil.rmtree(COMPILATION_CACHE_DIR)
    _remove_base_cache()


def _remove_base_cache():
    base = f"cachefiles/{DEVICE.upper()}"
    suffix = [".json", "-results.json", "-metadata.json"]
    for s in suffix:
        file = base + s
        if (os.path.exists(file)):
            os.remove(file)
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)

def _single_tune(use_compilation_cache: bool):
    start = time.perf_counter()
    results, env = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True, compilation_cache_enabled=use_compilation_cache)
    wall = time.perf_counter() - start
    return results, env, wall

def _extract_stats(results, env, wall_time):
    total_compile = env["total_compile_time"] / 1000.0
    total_benchmark = env["total_benchmark_time"] / 1000.0
    total_framework = env["total_framework_time"] / 1000.0
    total_strategy = env["total_strategy_time"] / 1000.0
    total_overhead = env["overhead_time"] / 1000.0

    cache_stats = env.get("compilation_cache_stats", 0)
    cache_hits = cache_stats.get("hits", 0)
    cache_misses = cache_stats.get("misses", 0)

    individual_compile_times = []
    for r in results:
        individual_compile_times.append(r["compile_time"])
    
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
        "cache_misses": cache_misses,
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
        "cache_hits",
        "cache_misses",
    ]
    aggregate = {}

    for key in keys:
        values = []
        for stat in stats:
            values.append(stat[key])
        aggregate[f"{key}_mean"] = np.mean(values)
        aggregate[f"{key}_std.dev"] = np.std(values, ddof = 1)
    

    return aggregate

def _calculate_speedup(cold, warm):
    def ratio(a, b):
        return a / b if b > 0 else float("NaN")
    
    return {
        "wallclock": ratio(cold["total_wall_clock_mean"], warm["total_wall_clock_mean"]),
        "compile": ratio(cold["total_compile_mean"], warm["total_compile_mean"])
    }


def _save_results(no_cache, cold, warm):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "device": DEVICE,
                "language": LANG,
                "cold": cold,
                "warm": warm,
                "speedup_no_vs cold": _calculate_speedup(no_cache, cold),
                "speedup_no_vs_warm": _calculate_speedup(no_cache, warm),
                "speedup_cold_vs_warm": _calculate_speedup(cold, warm)
            }, 
            f,
            indent=2
        )


def main():
    cold_stats, warm_stats, no_cache_stats = [], [], []

    for i in range(NUM_ITERATIONS):

        ## No cache
        _clear_cache()
        results, env, wall = _single_tune(use_compilation_cache=False)
        no_cache_stats.append(_extract_stats(results, env, wall))

        ## Cold cache
        _clear_cache()
        results, env, wall = _single_tune(use_compilation_cache=True)
        cold_stats.append(_extract_stats(results, env, wall))

        ## Warm cache
        _remove_base_cache()
        results, env, wall = _single_tune(use_compilation_cache=True)
        warm_stats.append(_extract_stats(results, env, wall))

    no_cache_aggregate = _run_N_times(no_cache_stats)
    cold_aggregate = _run_N_times(cold_stats)
    warm_aggregate = _run_N_times(warm_stats)

    _save_results(no_cache_aggregate, cold_aggregate, warm_aggregate)


main()