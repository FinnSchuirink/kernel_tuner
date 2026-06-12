import shutil
from experiments.parallel.convolution_milo.convolution_milo import tune
import os
import time
import json
import numpy as np

DEVICE = "A4000-Ada"
LANG = "CUDA"
NUM_ITERATIONS = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_LOC = os.path.join(BASE_DIR, "results", f"parallel_results_{DEVICE}_{LANG}.json")
PYCACHE = os.path.join(BASE_DIR, "__pycache__")

def _remove_base_cache():
    if (os.path.exists(PYCACHE)):
        shutil.rmtree(PYCACHE)

def _single_tune(runner_mode, num_threads=1):
    start = time.perf_counter()
    results, env = tune(device_name=DEVICE, lang=LANG, verbose=False, quiet=True, runner_mode=runner_mode, num_threads=num_threads)
    wall = time.perf_counter() - start
    return results, env, wall

def _extract_stats(env, wall_time, runner_mode):
    total_compile = env["total_compile_time"] / 1000.0 if runner_mode == "Sequential" else env["wall_compile_time"] / 1000.0
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

def _save_iter_results(iteration, iter_seq_results):
    RESULTS_ITER_LOC = os.path.join(BASE_DIR, "results", "iter_results", f"parallel_results_{DEVICE}_{LANG}.json")
    os.makedirs(os.path.dirname(RESULTS_ITER_LOC), exist_ok=True)

    with open(RESULTS_ITER_LOC, "w") as f:
        json.dump(
        {
            "Iteration": iteration,
            "Sequential": iter_seq_results,
        },
        f,
        indent=2
    )


def main():
    threaded_results = {}
    sequential_stats = []

    sequential_aggregate = {}

    seq_runner = "Sequential"

    for i in range(NUM_ITERATIONS):
        iter_results = {}
        _remove_base_cache()

        _, env, wall = _single_tune(runner_mode=seq_runner)
        iter_results[seq_runner] = _extract_stats(env, wall, seq_runner)

        sequential_stats.append(iter_results[seq_runner])

        sequential_aggregate = _run_N_times(sequential_stats)
        _save_iter_results(i + 1, sequential_aggregate)
    
    threaded_results["Sequential"] = sequential_aggregate

    _save_results(threaded_results)


main()