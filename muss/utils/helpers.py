# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from contextlib import contextmanager, AbstractContextManager
from fcntl import flock, LOCK_EX, LOCK_UN
from functools import wraps, lru_cache
import gzip
import hashlib
import inspect
from io import StringIO
from itertools import zip_longest
from pathlib import Path
import select
import shlex
import shutil
from subprocess import Popen, PIPE, CalledProcessError
import sys
import tempfile
import time
from types import MethodType

import cachetools
from imohash import hashfile
import numpy as np


def run_command(cmd, mute=False):
    def readline_with_timeout(input_stream, timeout=0.1):
        """Avoids handing indefinitely when calling readline()
        https://stackoverflow.com/questions/10756383/timeout-on-subprocess-readline-in-python"""
        poll_obj = select.poll()
        poll_obj.register(input_stream, select.POLLIN)
        start = time.time()
        while (time.time() - start) < timeout:
            poll_result = poll_obj.poll(0)
            if poll_result:
                return input_stream.readline()
        return ''

    def get_available_output(input_stream):
        output = ''
        while True:
            line = readline_with_timeout(input_stream, timeout=0.1)
            if line == '':
                break
            output += line
        return output

    def read_and_print(input_stream, output_stream):
        output = get_available_output(input_stream)
        if not mute:
            print(output, file=output_stream, end='', flush=True)
        return output

    # Inspired from subprocess.run() source
    # HACK: shell=True is not secure
    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as process:
        try:
            stdout = ''
            stderr = ''
            while True:
                stdout += read_and_print(process.stdout, sys.stdout)
                stderr += read_and_print(process.stderr, sys.stderr)
                if process.poll() is not None:
                    break
            # Read remaining output in case there is some
            stdout += read_and_print(process.stdout, sys.stdout)
            stderr += read_and_print(process.stderr, sys.stderr)
        except Exception:
            if mute:  # Print errors that could have been muted
                print(stderr, file=sys.stderr)
            print(get_available_output(process.stderr), file=sys.stderr)
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        if retcode:
            print(stderr, file=sys.stderr)
            raise CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
    return stdout.strip()


@contextmanager
def open_files(filepaths, mode='r'):
    files = []
    try:
        files = [Path(filepath).open(mode) for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [l.rstrip('\n') if l is not None else None for l in parallel_lines]
            yield parallel_lines


class FilesWrapper:
    '''Write to multiple open files at the same time'''

    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict  # Whether to raise an exception when a line is None

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip('\n') + '\n')


@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, 'w') as files:
        yield FilesWrapper(files, strict=strict)


def write_lines(lines, filepath=None):
    if filepath is None:
        filepath = get_temp_filepath()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        for line in lines:
            f.write(line + '\n')
    return filepath


def yield_lines(filepath, gzipped=False, n_lines=None):
    filepath = Path(filepath)
    open_function = open
    if gzipped or filepath.name.endswith('.gz'):
        open_function = gzip.open
    with open_function(filepath, 'rt') as f:
        for i, l in enumerate(f):
            if n_lines is not None and i >= n_lines:
                break
            yield l.rstrip('\n')


def read_lines(filepath, gzipped=False):
    return list(yield_lines(filepath, gzipped=gzipped))


def count_lines(filepath):
    n_lines = 0
    # We iterate over the generator to avoid loading the whole file in memory
    for _ in yield_lines(filepath):
        n_lines += 1
    return n_lines


@contextmanager
def open_with_lock(filepath, mode):
    with open(filepath, mode) as f:
        flock(f, LOCK_EX)
        yield f
        flock(f, LOCK_UN)


def get_lockfile_path(path):
    path = Path(path)
    if path.is_dir():
        return path / '.lockfile'
    if path.is_file():
        return path.parent / f'.{path.name}.lockfile'


@contextmanager
def lock_file(filepath):
    '''Lock file foo.txt by creating a lock on .foo.txt.lock'''
    # TODO: do we really need to create an additional file for locking ?
    filepath = Path(filepath)
    assert filepath.exists(), f'File does not exists: {filepath}'
    lockfile_path = get_lockfile_path(filepath)
    with open_with_lock(lockfile_path, 'w'):
        yield


@contextmanager
def lock_directory(dir_path):
    # TODO: Locking a directory should lock all files in that directory
    # Right now if we lock foo/, someone else can lock foo/bar.txt
    # TODO: Nested with lock_directory() should not be blocking
    assert Path(dir_path).exists(), f'Directory does not exists: {dir_path}'
    lockfile_path = get_lockfile_path(dir_path)
    with open_with_lock(lockfile_path, 'w'):
        yield


def failsafe_division(a, b, default=0):
    if b == 0:
        return default
    return a / b


def harmonic_mean(values, coefs=None):
    if 0 in values:
        return 0
    values = np.array(values)
    if coefs is None:
        coefs = np.ones(values.shape)
    values = np.array(values)
    coefs = np.array(coefs)
    return np.sum(coefs) / np.dot(coefs, 1 / values)


def arg_name_python_to_cli(arg_name, cli_sep='-'):
    arg_name = arg_name.replace('_', cli_sep)
    return f'--{arg_name}'


def arg_name_cli_to_python(arg_name, cli_sep='-'):
    assert arg_name.startswith('--')
    arg_name = arg_name.strip('-').replace(cli_sep, '_')
    return arg_name


def failsafe_ast_literal_eval(expression):
    try:
        return ast.literal_eval(expression.replace('PosixPath', ''))
    except (SyntaxError, ValueError):
        return expression


def cli_args_list_to_kwargs(cli_args_list):
    kwargs = {}
    i = 0
    while i < len(cli_args_list) - 1:
        assert cli_args_list[i].startswith('--'), cli_args_list[i]
        key = arg_name_cli_to_python(cli_args_list[i])
        next_element = cli_args_list[i + 1]
        if next_element.startswith('--'):
            kwargs[key] = True
            i += 1
        else:
            try:
                kwargs[key] = failsafe_ast_literal_eval(next_element)
            except (SyntaxError, ValueError):
                kwargs[key] = cli_args_list[i + 1]
            i += 2
    return kwargs


def kwargs_to_cli_args_list(kwargs, cli_sep='-'):
    cli_args_list = []
    for key, val in kwargs.items():
        key = arg_name_python_to_cli(key, cli_sep)
        if isinstance(val, bool):
            cli_args_list.append(str(key))
        else:
            if isinstance(val, str):
                # Add quotes around val
                assert "'" not in val
                val = f"'{val}'"
            cli_args_list.extend([str(key), str(val)])
    return cli_args_list


def args_str_to_dict(args_str):
    return cli_args_list_to_kwargs(shlex.split(args_str))


def args_dict_to_str(args_dict, cli_sep='-'):
    return ' '.join(kwargs_to_cli_args_list(args_dict, cli_sep=cli_sep))


@contextmanager
def redirect_streams(source_streams, target_streams):
    # We assign these functions before hand in case a target stream is also a source stream.
    # If it's the case then the write function would be patched leading to infinie recursion
    target_writes = [target_stream.write for target_stream in target_streams]
    target_flushes = [target_stream.flush for target_stream in target_streams]

    def patched_write(self, message):
        for target_write in target_writes:
            target_write(message)

    def patched_flush(self):
        for target_flush in target_flushes:
            target_flush()

    original_source_stream_writes = [source_stream.write for source_stream in source_streams]
    original_source_stream_flushes = [source_stream.flush for source_stream in source_streams]
    try:
        for source_stream in source_streams:
            source_stream.write = MethodType(patched_write, source_stream)
            source_stream.flush = MethodType(patched_flush, source_stream)
        yield
    finally:
        for source_stream, original_source_stream_write, original_source_stream_flush in zip(
            source_streams, original_source_stream_writes, original_source_stream_flushes
        ):
            source_stream.write = original_source_stream_write
            source_stream.flush = original_source_stream_flush


@contextmanager
def mute(mute_stdout=True, mute_stderr=True):
    streams = []
    if mute_stdout:
        streams.append(sys.stdout)
    if mute_stderr:
        streams.append(sys.stderr)
    with redirect_streams(source_streams=streams, target_streams=StringIO()):
        yield


@contextmanager
def log_std_streams(filepath):
    log_file = open(filepath, 'w')
    try:
        with redirect_streams(source_streams=[sys.stdout], target_streams=[log_file, sys.stdout]):
            with redirect_streams(source_streams=[sys.stderr], target_streams=[log_file, sys.stderr]):
                yield
    finally:
        log_file.close()


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


@lru_cache(maxsize=10000)
def get_file_hash(filepath):
    return hashfile(filepath, hexdigest=True)


def get_string_hash(string):
    return hashlib.md5(string.encode()).hexdigest()


def get_files_hash(filepaths):
    return get_string_hash(''.join([get_file_hash(path) for path in sorted(filepaths)]))


class SkipWithBlock(Exception):
    pass


class create_directory_or_skip(AbstractContextManager):
    """Context manager for creating a new directory (with rollback and skipping with block if exists)

    In order to skip the execution of the with block if the dataset already exists, this context manager uses deep
    magic from https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
    """

    def __init__(self, dir_path, overwrite=False):
        self.dir_path = Path(dir_path)
        self.overwrite = overwrite

    def __enter__(self):
        if self.dir_path.exists():
            self.directory_lock = lock_directory(self.dir_path)
            self.directory_lock.__enter__()
            files_in_directory = list(self.dir_path.iterdir())
            if set(files_in_directory) in [set([]), set([self.dir_path / '.lockfile'])]:
                # TODO: Quick hack to remove empty directories
                self.directory_lock.__exit__(None, None, None)
                print(f'Removing empty directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
            else:
                # Deep magic hack to skip the execution of the code inside the with block
                # We set the trace to a dummy function
                sys.settrace(lambda *args, **keys: None)
                # Get the calling frame (sys._getframe(0) is the current frame)
                frame = sys._getframe(1)
                # Set the calling frame's trace to the one that raises the special exception
                frame.f_trace = self.trace
                return
        print(f'Creating "{self.dir_path}"...')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.directory_lock = lock_directory(self.dir_path)
        self.directory_lock.__enter__()

    def trace(self, frame, event, arg):
        # This function is called when a new local scope is entered, i.e. right when the code in the with block begins
        # The exception will therefore be caught by the __exit__()
        raise SkipWithBlock()

    def __exit__(self, type, value, traceback):
        self.directory_lock.__exit__(type, value, traceback)
        if type is not None:
            if issubclass(type, SkipWithBlock):
                return True  # Suppress special SkipWithBlock exception
            if issubclass(type, BaseException):
                # Rollback
                print(f'Error: Rolling back creation of directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
                return False  # Reraise the exception


TEMP_DIR = None


def get_temp_filepath(create=False):
    global TEMP_DIR
    temp_filepath = Path(tempfile.mkstemp()[1])
    if TEMP_DIR is not None:
        temp_filepath.unlink()
        temp_filepath = TEMP_DIR / temp_filepath.name
        temp_filepath.touch(exist_ok=False)
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def get_temp_filepaths(n_filepaths, create=False):
    return [get_temp_filepath(create=create) for _ in range(n_filepaths)]


def get_temp_dir():
    return Path(tempfile.mkdtemp())


@contextmanager
def create_temp_dir():
    temp_dir = get_temp_dir()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def delete_files(filepaths):
    for filepath in filepaths:
        filepath = Path(filepath)
        assert filepath.is_file()
        filepath.unlink()


@contextmanager
def log_action(action_description):
    start_time = time.time()
    print(f'{action_description}...')
    try:
        yield
    except BaseException as e:
        print(f'{action_description} failed after {time.time() - start_time:.2f}s.')
        raise e
    print(f'{action_description} completed after {time.time() - start_time:.2f}s.')


def print_running_time(func):
    '''Decorator to print running time of function for logging purposes'''

    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        function_name = getattr(func, '__name__', repr(func))
        with log_action(function_name):
            return func(*args, **kwargs)

    return wrapped_func


def get_hashable_object(obj):
    def get_hashable_dict(d):
        return tuple(sorted([(key, get_hashable_object(value)) for key, value in d.items()]))

    def get_hashable_list(l):
        return tuple(l)

    def get_hashable_numpy_array(arr):
        # Note: tobytes() Makes a copy of the array
        return arr.tobytes()

    if isinstance(obj, list):
        return get_hashable_list(obj)
    if isinstance(obj, dict):
        return get_hashable_dict(obj)
    if isinstance(obj, np.ndarray):
        return get_hashable_numpy_array(obj)
    return obj


def generalized_lru_cache(maxsize=128):
    '''Decorator factory'''

    def _generalized_lru_cache(function):
        '''Actual decorator'''

        def hash_keys(*args, **kwargs):
            def generalized_hash(arg):
                '''Hashes objects that are not hashable by default'''
                return hash(get_hashable_object(arg))

            args = tuple(generalized_hash(arg) for arg in args)
            kwargs = {k: generalized_hash(v) for k, v in kwargs.items()}
            return cachetools.keys.hashkey(*args, **kwargs)

        return cachetools.cached(
            cache=cachetools.LRUCache(maxsize=maxsize),
            key=hash_keys,
        )(function)

    return _generalized_lru_cache


def batch_items(item_generator, batch_size):
    batch = []
    for item in item_generator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


@contextmanager
def mock_cli_args(args):
    current_args = sys.argv
    sys.argv = sys.argv[:1] + args
    yield
    sys.argv = current_args
