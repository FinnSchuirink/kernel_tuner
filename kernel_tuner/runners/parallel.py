"""The multithreaded runner for parallel tuning of the parameter space."""
import logging
import os
from datetime import datetime, timezone
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from kernel_tuner.core import DeviceInterface
from kernel_tuner.runners.runner import Runner
from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache

class ParallelRunner(Runner):
    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers, num_threads=1, compilation_cache_enabled=False):
        """Instantiate the ParallelRunner.

            :param kernel_source: The kernel source
            :type kernel_source: kernel_tuner.core.KernelSource

            :param kernel_options: A dictionary with all options for the kernel.
            :type kernel_options: kernel_tuner.interface.Options

            :param device_options: A dictionary with all options for the device
                on which the kernel should be tuned.
            :type device_options: kernel_tuner.interface.Options

            :param iterations: The number of iterations used for benchmarking
                each kernel instance.
            :type iterations: int
        """
        #detect language and create high-level device interface
        self.dev = DeviceInterface(kernel_source, compilation_cache_enabled=compilation_cache_enabled, iterations=iterations, observers=observers, **device_options)

        self.cuda_context = None
        # Save current CUDA context to be pushed to threads later
        try:
            import pycuda.driver as drv
            self.cuda_context = drv.Context.get_current()
        except:
            pass

        self.units = self.dev.units
        self.quiet = device_options.quiet
        self.kernel_source = kernel_source
        self.warmed_up = False if self.dev.requires_warmup else True
        self.runner_mode = "Parallel"
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
        self.kernel_options = kernel_options
        self.num_threads = num_threads

        #move data to the GPU
        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)
        self.wall_compile_time = 0

    def get_environment(self, tuning_options):
        env = self.dev.get_environment()
        env["wall_compile_time"] = self.wall_compile_time
        return env

    def single_compilation(self, element, tuning_options):
        """

            :param element: The current configuration inside the parameter space currently being observed
            :type element: tuple

            :param tuning_options: A dictionary with all options regarding the tuning process.
            :type tuning_options: kernel_tuner.interface.Options

            :returns tuple of necessary information for benchmarking:
        """
        ## Copy cuda_context onto the thread
        if (self.cuda_context is not None):
            self.cuda_context.push()
        try:
            ## Build params dictionary
            params = dict(zip(tuning_options.tune_params.keys(), element))

            result, func, to, instance = self.dev.compile(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)
        finally:
            # Ensures removal of the cuda_context from the thread if compilation fails
            if (self.cuda_context is not None):
                self.cuda_context.pop()

        return result, func, to, instance

    def run(self, parameter_space, tuning_options):
        """

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.interface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times.
        :rtype: dict())

        """
        logging.debug('parallel runner started for ' + self.kernel_options.kernel_name)

        results = []
        warmup_time = 0

        # Results for the individual threads
        future_results = {}

        # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
        if not self.warmed_up:
            first = next(iter(parameter_space))
            params = dict(zip(tuning_options.tune_params.keys(), first))
            warmup_time = perf_counter()
            self.dev.compile(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)
            self.warmed_up = True
            warmup_time = 1e3 * (perf_counter() - warmup_time)

        # iterate over parameter space using thread pool
        max_workers = self.num_threads if self.num_threads > 1 else 1

        compile_wall_start = perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as batch_executor:

            cached_results = []

            # Queue all tasks
            for element in parameter_space:
                x_int = ",".join([str(i) for i in element])

                # Check if result is already in the cache
                if tuning_options.cache and x_int in tuning_options.cache:
                    params = dict(zip(tuning_options.tune_params.keys(), element))
                    params.update(tuning_options.cache[x_int])
                    params['compile_time'] = 0
                    params['verification_time'] = 0
                    params['benchmark_time'] = 0
                    cached_results.append(params)

                else:
                    future_results[batch_executor.submit(self.single_compilation, element, tuning_options)] = element

            ## Wait for all configurations to finish compiling
            wait(future_results)
            self.wall_compile_time = perf_counter() - compile_wall_start

            # Collect results as they were completed
            for future_result in as_completed(future_results):
                result = None
                element = future_results[future_result]
                x_int = ",".join([str(i) for i in element])
                params = dict(zip(tuning_options.tune_params.keys(), element))

                # Retrieve the result
                result, func, to, instance = future_result.result()

                # Only benchmark if the compilation succeeded
                if func is not None:
                    benchmark_result = self.dev.benchmark_kernel(instance, func, self.gpu_args, to, result)

                    # Save benchmark results
                    result.update(benchmark_result)

                # Map result to the parameter
                params.update(result)

                if tuning_options.objective in result and isinstance(result[tuning_options.objective], ErrorConfig):
                    logging.debug('kernel configuration was skipped silently due to compile or runtime failure')

                # only compute metrics on configs that have not errored
                if tuning_options.metrics and not isinstance(params.get(tuning_options.objective), ErrorConfig):
                    params = process_metrics(params, tuning_options.metrics)

                # get the framework time by estimating based on other times
                total_time = 1000 * ((perf_counter() - self.start_time) - warmup_time)
                params['compile_time'] = self.wall_compile_time
                params['strategy_time'] = self.last_strategy_time
                params['framework_time'] = max(total_time - (params['compile_time'] + params['verification_time'] + params['benchmark_time'] + params['strategy_time']), 0)   
                params['timestamp'] = str(datetime.now(timezone.utc))
                self.start_time = perf_counter()

                if result:
                    # print configuration to the console
                    print_config_output(tuning_options.tune_params, params, self.quiet, tuning_options.metrics, self.units)

                    # add configuration to cache
                    store_cache(x_int, params, tuning_options)

                # all visited configurations are added to results to provide a trace for optimization strategies
                results.append(params)
            results.extend(cached_results)

        return results