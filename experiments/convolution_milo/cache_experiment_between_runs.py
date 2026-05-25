import shutil
from experiments.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "GTX1650"
LANG = "CUDA"
NUM_ITERATIONS = 5
RESULTS_LOC = f"results/cache_experiment_between_runs_{DEVICE}_{LANG}.json"
PYCACHE = "__pycache__"

def _clear_cache(cache_dir="compilation_cache"):
    if (os.path.exists(cache_dir)):
        shutil.rmtree(cache_dir)
    _remove_base_cache()


def _remove_base_cache():
    base = f"cachefiles/{DEVICE}"
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

def _calculate_speedup(cold_mean, warm_mean):
    if warm_mean == 0:
        return float("NaN"), float("NaN")
    
    speedup = cold_mean / warm_mean
    return speedup


def _save_results(cold: list, warm: list):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "device": DEVICE,
                "language": LANG,
                "cold": cold,
                "warm": warm,
                "speedup": _calculate_speedup(cold["total_compile_mean"], warm["total_compile_mean"])
            }, 
            f,
            indent=2
        )


def main():
    cold_stats, warm_stats = [], []

    for i in range(NUM_ITERATIONS):
        _clear_cache()
        results, env, wall = _single_tune(use_compilation_cache=True)
        cold_stats.append(_extract_stats(results, env, wall))

        _remove_base_cache()

        results, env, wall = _single_tune(use_compilation_cache=True)
        warm_stats.append(_extract_stats(results, env, wall))

    cold_aggregate = _run_N_times(cold_stats)
    warm_aggregate = _run_N_times(warm_stats)

    _save_results(cold_aggregate, warm_aggregate)


main()