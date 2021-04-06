# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
import traceback

import numpy as np
from submitit import AutoExecutor
from submitit.helpers import DelayedSubmission
from submitit.core.job_environment import JobEnvironment

from muss.resources.paths import SUBMITIT_JOB_DIR_FORMAT, EXP_DIR
from muss.utils.helpers import print_running_time, generalized_lru_cache
from muss.utils.training import print_function_name, print_args, print_result


def get_job_id():
    try:
        return JobEnvironment().job_id
    except RuntimeError:
        return None


def make_function_checkpointable(function):
    def checkpoint(*args, **kwargs):
        return DelayedSubmission(function, *args, **kwargs)  # submits to requeuing

    function.checkpoint = checkpoint
    return function


def print_job_id(func):
    '''Decorator to print slurm id for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        print(f'job_id={get_job_id()}')
        return func(*args, **kwargs)

    return wrapped_func


@generalized_lru_cache()
def get_executor(
    cluster='slurm',
    submit_decorators=[
        make_function_checkpointable,
        print_function_name,
        print_args,
        print_job_id,
        print_result,
        print_running_time,
    ],
    timeout_min=1 * 12 * 60,
    slurm_partition='learnfair',
    catch=False,
    gpus_per_node=1,
    cpus_per_task=None,
    nodes=1,
    mem_gb=None,
    slurm_max_num_timeout=3,
    **kwargs,
):
    assert gpus_per_node <= 8
    executor = AutoExecutor(
        folder=SUBMITIT_JOB_DIR_FORMAT, cluster=cluster, slurm_max_num_timeout=slurm_max_num_timeout
    )
    if catch:
        executor = executor_with_catch(executor)
    if cpus_per_task is None:
        cpus_per_task = gpus_per_node * 10
    if mem_gb is None:
        mem_gb = gpus_per_node * 64
    executor.update_parameters(
        timeout_min=timeout_min,
        slurm_partition=slurm_partition,
        gpus_per_node=gpus_per_node,
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        mem_gb=mem_gb,
        **kwargs,
    )
    for decorator in submit_decorators:
        executor.submit = get_decorated_submit(executor.submit, decorator)
    return executor


def get_decorated_submit(submit, decorator):
    @wraps(submit)
    def decorated_submit(func, *args, **kwargs):
        decorated_func = decorator(func)
        if hasattr(decorated_func, 'checkpoint'):
            decorated_func.checkpoint = func.checkpoint  # Transfer the checkpoint function
        return submit(decorated_func, *args, **kwargs)

    return decorated_submit


# Catch job failed for nevergrad and return a score of 1
def result_with_catch(result):
    @wraps(result)
    def catched_result():
        try:
            score = result()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            filepath = EXP_DIR / 'nevergrad_failed'
            with open(filepath, 'a') as f:
                f.write(f'{get_job_id()}: {e}\n')
            score = np.inf
        return score

    return catched_result


def job_with_catch(job):
    job.result = result_with_catch(job.result)
    return job


def submit_with_catch(submit):
    @wraps(submit)
    def catched_submit(func, *args, **kwargs):
        job = submit(func, *args, **kwargs)
        return job_with_catch(job)

    return catched_submit


def executor_with_catch(executor):
    executor.submit = submit_with_catch(executor.submit)
    return executor
