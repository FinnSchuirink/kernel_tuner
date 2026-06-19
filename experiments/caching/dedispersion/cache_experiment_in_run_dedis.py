import shutil
from experiments.caching.pnpoly.pnpoly import tune
import os
import time
import json
import numpy as np

NUM_ITERATIONS = 10

COMPILE = str(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_LOC = os.path.join(BASE_DIR, COMPILE, "results", "cache_experiment_in_run.json")
RESULTS_ITER_LOC = os.path.join(BASE_DIR, COMPILE, "results", "iter_results", "cache_results_in_run.json")
PYCACHE = os.path.join(BASE_DIR, COMPILE, "__pycache__")
COMPILATION_CACHE_DIR = os.path.join(BASE_DIR, COMPILE, "compilation_cache")

def _clear_cache():
    if (os.path.exists(COMPILATION_CACHE_DIR)):
        shutil.rmtree(COMPILATION_CACHE_DIR)
    _remove_base_cache()

def _remove_base_cache():
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)
    
def _single_tune(use_compilation_cache: bool):
    start = time.perf_counter()
    results, env = tune(compilation_cache_enabled=use_compilation_cache)
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
        aggregate[f"{key}_std.dev"] = float(np.std(values, ddof = 1)) if len(values) > 1 else float("NaN")
    
    return aggregate

def _calculate_speedup(cold, warm):
    def ratio(a, b):
        return a / b if b > 0 else float("NaN")
    
    return {
        "wallclock": ratio(cold.get("total_wallclock_mean", 0), warm.get("total_wallclock_mean", 0)),
        "compile": ratio(cold.get("total_compile_mean", 0), warm.get("total_compile_mean", 0))
    }


def _save_results(no_cache, cache):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "no_cache": no_cache,
                "warm": cache,
                "speedup_no_cache_vs_cache": _calculate_speedup(no_cache, cache),
            }, 
            f,
            indent=2
        )

def _save_iter_results(iteration, iter_no_cache_results, iter_cache_results):
    os.makedirs(os.path.dirname(RESULTS_ITER_LOC), exist_ok=True)

    with open(RESULTS_ITER_LOC, "w") as f:
        json.dump(
        {
            "Iteration": iteration,
            "no_cache": iter_no_cache_results,
            "warm": iter_cache_results,
            "speedup_no_cache_vs_cache": _calculate_speedup(iter_no_cache_results, iter_cache_results),
        },
        f,
        indent=2
    )

def main():
    cache_stats, no_cache_stats = [], []
    no_cache_aggregate, cache_aggregate = {}, {}

    for i in range(NUM_ITERATIONS):
        print(f"Iteration {i + 1}/{NUM_ITERATIONS}", flush=True)
        ## No cache
        _clear_cache()
        results, env, wall = _single_tune(use_compilation_cache=False)
        no_cache_stats.append(_extract_stats(results, env, wall))
        no_cache_aggregate = _run_N_times(no_cache_stats)

        _save_iter_results(i + 1, no_cache_aggregate, cache_aggregate)

        ## Cache
        _clear_cache()
        results, env, wall = _single_tune(use_compilation_cache=True)
        cache_stats.append(_extract_stats(results, env, wall))
        cache_aggregate = _run_N_times(cache_stats)

        _save_iter_results(i + 1, no_cache_aggregate, cache_aggregate)

    _save_results(no_cache_aggregate, cache_aggregate)


main()