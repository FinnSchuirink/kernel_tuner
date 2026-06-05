import shutil
from experiments.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "A4000-Ada"
LANG = "CUDA"
NUM_ITERATIONS = 10
RESULTS_LOC = f"results/cache_experiment_between_runs_{DEVICE}_{LANG}.json"
RESULTS_ITER_LOC = f"results/iter_results/caching/cache_results_between_runs.json"
PYCACHE = "__pycache__"
COMPILATION_CACHE_DIR = "compilation_cache"
BENCHMARK_CACHE = "pnpoly_cache.json"

def _clear_cache():
    if (os.path.exists(COMPILATION_CACHE_DIR)):
        shutil.rmtree(COMPILATION_CACHE_DIR)
    _remove_base_cache()


def _remove_base_cache():
    base = f"cachefiles/{DEVICE.upper()}"
    for suffix in [".json", "-results.json", "-metadata.json"]:
        path = base + suffix
        if (os.path.exists(path)):
            os.remove(path)
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)
    if (os.path.exists(BENCHMARK_CACHE)):
        os.remove(BENCHMARK_CACHE)

def _single_tune(use_compilation_cache):
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

    cache_stats = env.get("compilation_cache_stats", {})
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
        "n_configs",
    ]
    aggregate = {}

    for key in keys:
        values = []
        for stat in stats:
            values.append(stat[key])
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std.dev"] = (float(np.std(values, ddof = 1)) if len(values) > 1 else float("NaN"))
    
    return aggregate

def _calculate_speedup(cold, warm):
    def ratio(a, b):
        return a / b if b > 0 else float("NaN")
    
    return {
        "wallclock": ratio(cold["total_wallclock_mean"], warm["total_wallclock_mean"]),
        "compile": ratio(cold["total_compile_mean"], warm["total_compile_mean"])
    }


def _save_results(no_cache, cold, warm):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "device": DEVICE,
                "language": LANG,
                "no_cache": no_cache,
                "cold": cold,
                "warm": warm,
                "speedup_no_vs_cold": _calculate_speedup(no_cache, cold),
                "speedup_no_vs_warm": _calculate_speedup(no_cache, warm),
                "speedup_cold_vs_warm": _calculate_speedup(cold, warm)
            }, 
            f,
            indent=2
        )

def _save_iter_results(iteration, iter_no_cache_results, iter_cold_cache_results, iter_warm_cache_results):
    os.makedirs(os.path.dirname(RESULTS_ITER_LOC), exist_ok=True)

    with open(RESULTS_ITER_LOC, "w") as f:
        json.dump(
        {
            "Iteration": iteration,
            "no_cache": iter_no_cache_results,
            "cold": iter_cold_cache_results,
            "warm": iter_warm_cache_results,
            "speedup_no_vs_cold": _calculate_speedup(iter_no_cache_results, iter_cold_cache_results),
            "speedup_no_vs_warm": _calculate_speedup(iter_no_cache_results, iter_warm_cache_results),
            "speedup_cold_vs_warm": _calculate_speedup(iter_cold_cache_results, iter_warm_cache_results)
        },
        f,
        indent=2
    )


def main():
    cold_stats, warm_stats, no_cache_stats = [], [], []
    no_cache_aggregate, cold_aggregate, warm_aggregate = {}, {}, {}

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i + 1}/{NUM_ITERATIONS}", flush=True)

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

        _save_iter_results(i + 1, no_cache_aggregate, cold_aggregate, warm_aggregate)

    _save_results(no_cache_aggregate, cold_aggregate, warm_aggregate)


main()