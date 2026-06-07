import shutil
from experiments.parallel.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "A4000-Ada"
LANG = "CUDA"
NUM_ITERATIONS = 3
NUM_THREADS = [1, 2, 4, 8, 16, 32, 64]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_LOC = os.path.join(BASE_DIR, "results", f"parallel_results_{DEVICE}_{LANG}.json")
PYCACHE = os.path.join(BASE_DIR, "__pycache__")
KT_CACHE = os.path.join(BASE_DIR, DEVICE.upper())

def _remove_base_cache():
    base = KT_CACHE
    for suffix in [".json", "-Sequential-results.json", "-Parallel-results.json", "-Parallel-metadata.json", "-Sequential-metadata.json"]:
        path = base + suffix
        if (os.path.exists(path)):
            os.remove(path)
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)

def _single_tune(runner_mode, num_threads=1):
    start = time.perf_counter()
    results, env = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True, runner_mode=runner_mode, num_threads=num_threads)
    wall = time.perf_counter() - start
    return results, env, wall

def _extract_stats(env, wall_time, runner_mode):
    total_compile = env["total_compile_time"] / 1000.0 if runner_mode == "Sequential" else env["wall_compile_time"]
    total_benchmark = env["total_benchmark_time"] / 1000.0
    total_framework = env["total_framework_time"] / 1000.0
    total_strategy = env["total_strategy_time"] / 1000.0
    total_overhead = wall_time - total_compile - total_benchmark - total_framework - total_strategy

    return {
        "total_compile": total_compile,
        "total_benchmark": total_benchmark,
        "total_framework": total_framework,
        "total_strategy": total_strategy,
        "total_overhead": total_overhead,

        "total_wallclock": wall_time,
        "compile_fraction": total_compile / wall_time if wall_time > 0 else float("NaN"),
       }

def _run_N_times(stats):
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
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std.dev"] = (
            float(np.std(values, ddof = 1)) if len(values) > 1 else float("NaN")
        )

    return aggregate

def _calculate_speedup(seq_stats, parallel_stats):
    def ratio(a, b):
        return a / b if b > 0 else float("NaN")

    return {
        "wallclock": ratio(seq_stats["total_wallclock_mean"], parallel_stats["total_wallclock_mean"]),
        "compile": ratio(seq_stats["total_compile_mean"], parallel_stats["total_compile_mean"])
    }


def _save_results(threaded_results):
    os.makedirs(os.path.dirname(RESULTS_LOC), exist_ok=True)
    with open(RESULTS_LOC, "w") as f:
        json.dump(
            {
                "device": DEVICE,
                "language": LANG,
                "results": threaded_results,
            },
            f,
            indent=2
        )

def _save_iter_results(iteration, iter_seq_results, iter_parallel_results, num_threads):
    RESULTS_ITER_LOC = os.path.join(BASE_DIR, "results", "iter_results", f"{num_threads}", f"parallel_results_{DEVICE}_{LANG}.json")
    os.makedirs(os.path.dirname(RESULTS_ITER_LOC), exist_ok=True)

    with open(RESULTS_ITER_LOC, "w") as f:
        json.dump(
        {
            "Iteration": iteration,
            "Sequential": iter_seq_results,
            "Parallel": iter_parallel_results,
            "Speedup": _calculate_speedup(iter_seq_results, iter_parallel_results)
        },
        f,
        indent=2
    )


def main():
    threaded_results = {}
    sequential_stats = []

    seq_runner = "Sequential"
    parallel_runner = "Parallel"

    for i in range(NUM_ITERATIONS):
        iter_results = {}
        _remove_base_cache()

        _, env, wall = _single_tune(runner_mode=seq_runner)
        iter_results[seq_runner] = _extract_stats(env, wall, seq_runner)

        sequential_stats.append(iter_results[seq_runner])

    sequential_aggregate = _run_N_times(sequential_stats)

    for num_threads in NUM_THREADS:
        parallel_stats = []

        for i in range(NUM_ITERATIONS):
            print(f"Thread {num_threads}: Iteration {i + 1}/{NUM_ITERATIONS}", flush = True)

            iter_results = {}
            _remove_base_cache()

            _, env, wall = _single_tune(runner_mode=parallel_runner, num_threads=num_threads)
            iter_results[parallel_runner] = _extract_stats(env, wall, parallel_runner)

            parallel_stats.append(iter_results[parallel_runner])
            parallel_aggregate = _run_N_times(parallel_stats)

            _save_iter_results(i + 1, sequential_aggregate, parallel_aggregate, num_threads)

        parallel_aggregate = _run_N_times(parallel_stats)
    
        threaded_results[num_threads] = {
            "NUM_THREADS": num_threads,
            "Sequential": sequential_aggregate,
            "Parallel": parallel_aggregate,
            "Speedup": _calculate_speedup(sequential_aggregate, parallel_aggregate)
        }

    _save_results(threaded_results)


main()