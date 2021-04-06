# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from itertools import product
from pathlib import Path
import shutil

import numpy as np

from muss.preprocessing import shuffle_file_lines, get_parallel_file_pair_preprocessor, apply_line_function_to_file
from muss.preprocessors import dump_preprocessors, load_preprocessors
from muss.resources.paths import DATASETS_DIR, PHASES, LANGUAGES, get_dataset_dir, get_data_filepath, get_filepaths_dict
from muss.utils.helpers import count_lines, yield_lines, read_lines, create_directory_or_skip


def yield_indexes_of_lines(filepath, lines):
    lines = set(lines)
    with Path(filepath).open('r') as f:
        for idx, line in enumerate(f):
            if line.strip('\n') in lines:
                yield idx


def sort_files_by_line_count(filepaths):
    return sorted(filepaths, key=lambda filepath: count_lines(filepath))


def has_lines_in_common(filepath1, filepath2):
    [smallest_filepath, largest_filepath] = sort_files_by_line_count([filepath1, filepath2])
    for idx in yield_indexes_of_lines(largest_filepath, read_lines(smallest_filepath)):
        return True
    return False


def create_smaller_dataset(dataset, n_lines):
    new_dataset = f'{dataset}-lines{n_lines}'
    with create_directory_or_skip(get_dataset_dir(new_dataset)):
        filepaths_dict = get_filepaths_dict(dataset)
        new_filepaths_dict = get_filepaths_dict(new_dataset)
        for phase, language in product(['train'], LANGUAGES):
            with open(new_filepaths_dict[(phase, language)], 'w') as output_file:
                for line in yield_lines(filepaths_dict[(phase, language)], n_lines=n_lines):
                    output_file.write(line + '\n')
        for phase, language in product(['valid', 'test'], LANGUAGES):
            shutil.copy(filepaths_dict[(phase, language)], new_filepaths_dict[(phase, language)])
    return new_dataset


def mix_files(input_filepaths, props, output_filepath):
    np.random.seed(0)
    generators = [yield_lines(filepath) for filepath in input_filepaths]
    has_looped = [False] * len(input_filepaths)
    # Stop when all lines have been seen at least once
    with open(output_filepath, 'w') as f:
        while True:
            idx = np.random.choice(range(len(input_filepaths)), p=props)
            try:
                line = next(generators[idx])
            except StopIteration:
                has_looped[idx] = True
                # Start reading the file all over again
                generators[idx] = yield_lines(input_filepaths[idx])
                line = next(generators[idx])
            if all(has_looped):
                break
            f.write(f'{line}\n')


def mix_datasets(datasets, props=None, new_dataset=None):
    if len(set(datasets)) == 1:
        return datasets[0]
    if props is None:
        props = [1 / len(datasets)] * len(datasets)
    assert len(props) == len(datasets)
    assert all([get_dataset_dir(dataset).exists() for dataset in datasets])
    # Sort in unison according to dataset names
    datasets, props = zip(*sorted(zip(datasets, props)))
    if new_dataset is None:
        new_dataset = 'mix-' + '-'.join([f'{dataset}_{prop:.2f}' for dataset, prop in zip(datasets, props)])
    with create_directory_or_skip(get_dataset_dir(new_dataset)):
        print('Mixing datasets...')
        for phase, language in product(PHASES, LANGUAGES):
            input_files = [get_data_filepath(dataset, phase, language) for dataset in datasets]
            # If one of the input files does not exist, we remove it and its prop and renormalize
            input_files, current_props = zip(
                *[(input_file, prop) for input_file, prop in zip(input_files, props) if input_file.exists()]
            )
            current_props = np.array(current_props) / np.sum(current_props)
            output_file = get_data_filepath(new_dataset, phase, language)
            # TODO: Jointly mix files
            # The seed is set everytime mix is called, therefore they should be mixed in the same order
            mix_files(input_files, current_props, output_file)
            shuffle_file_lines(output_file)
    return new_dataset


def get_preprocessed_dataset_name(dataset, preprocessor):
    return '_' + hashlib.md5((dataset + preprocessor.get_hash()).encode()).hexdigest()


def create_preprocessed_dataset_one_preprocessor(dataset, preprocessor, n_jobs):
    new_dataset = get_preprocessed_dataset_name(dataset, preprocessor)
    with create_directory_or_skip(get_dataset_dir(new_dataset)):
        print(f'Creating preprocessed dataset with {preprocessor}: {dataset} -> {new_dataset}')
        new_dataset_dir = get_dataset_dir(new_dataset)
        filepaths_dict = get_filepaths_dict(dataset)
        new_filepaths_dict = get_filepaths_dict(new_dataset)
        for phase in PHASES:
            if not filepaths_dict[phase, 'complex'].exists() or not filepaths_dict[phase, 'complex'].exists():
                continue
            parallel_file_pair_preprocessor = get_parallel_file_pair_preprocessor(
                preprocessor.encode_file_pair,
                n_jobs=n_jobs,
            )
            parallel_file_pair_preprocessor(
                filepaths_dict[phase, 'complex'],
                filepaths_dict[phase, 'simple'],
                new_filepaths_dict[phase, 'complex'],
                new_filepaths_dict[phase, 'simple'],
            )
        previous_preprocessors = load_preprocessors(get_dataset_dir(dataset))
        if previous_preprocessors is not None:
            preprocessors = previous_preprocessors + [preprocessor]
        else:
            preprocessors = [preprocessor]
        dump_preprocessors(preprocessors, new_dataset_dir)
        with open(new_dataset_dir / 'original_dataset', 'w') as f:
            f.write(dataset + '\n')
        if hasattr(preprocessor, 'copy_sentencepiece_files_to_dir'):
            preprocessor.copy_sentencepiece_files_to_dir(new_dataset_dir)
    return new_dataset


def create_preprocessed_dataset(dataset, preprocessors, n_jobs=1):
    for preprocessor in preprocessors:
        # Fit preprocessor on input dataset
        preprocessor.fit(get_data_filepath(dataset, 'train', 'complex'), get_data_filepath(dataset, 'train', 'simple'))
        dataset = create_preprocessed_dataset_one_preprocessor(dataset, preprocessor, n_jobs)
    return dataset


def get_all_datasets():
    return [dir_path.name for dir_path in DATASETS_DIR.iterdir() if dir_path.is_dir()]


def get_original_dataset(dataset):
    filepath = get_dataset_dir(dataset) / 'original_dataset'
    if not filepath.exists():
        return None
    [original_dataset] = read_lines(filepath)
    return original_dataset


def get_downstream_preprocessed_datasets(dataset):
    downstream_datasets = []
    for candidate_dataset in get_all_datasets():
        original_dataset = get_original_dataset(candidate_dataset)
        if original_dataset is not None and original_dataset == dataset:
            downstream_datasets.append(candidate_dataset)
    # Now search recursively
    for downstream_dataset in downstream_datasets.copy():
        downstream_datasets.extend(get_downstream_preprocessed_datasets(downstream_dataset))
    return downstream_datasets


def get_upstream_preprocessed_datasets(dataset):
    original_dataset = get_original_dataset(dataset)
    if original_dataset is None:
        return []
    return [original_dataset] + get_upstream_preprocessed_datasets(original_dataset)


def apply_line_function_to_dataset(line_function, dataset, new_dataset, languages=LANGUAGES):
    '''Provided function signature: line_function(line) -> line'''
    with create_directory_or_skip(get_dataset_dir(new_dataset)):
        for phase, language in product(PHASES, languages):
            source_filepath = get_data_filepath(dataset, phase, language)
            target_filepath = get_data_filepath(new_dataset, phase, language)
            if not source_filepath.exists():
                continue
            apply_line_function_to_file(line_function, source_filepath, target_filepath)
    return new_dataset
