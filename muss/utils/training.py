# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

import torch


def print_function_name(func):
    '''Decorator to print method name for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        function_name = getattr(func, '__name__', repr(func))
        print(f"function_name='{function_name}'")
        return func(*args, **kwargs)

    return wrapped_func


def print_args(func, collapse=True):
    '''Decorator to print arguments of function for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        args_str = str(args)
        kwargs_str = str(kwargs)
        if collapse:
            max_length = 1000
            if len(args_str) > max_length:
                args_str = f'{args_str[:1000]}...'
            if len(kwargs_str) > max_length:
                kwargs_str = f'{kwargs_str[:1000]}...'
        print(f'args={args_str}')
        print(f'kwargs={kwargs_str}')
        return func(*args, **kwargs)

    return wrapped_func


def print_result(func):
    '''Decorator to print result of function for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'result={result}')
        return result

    return wrapped_func


def clear_cuda_cache(func):
    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.empty_cache()

    return wrapped_func
