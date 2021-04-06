# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
from pathlib import Path
import shutil

from imohash import hashfile

from muss.fairseq.base import fairseq_generate
from muss.preprocessors import ComposedPreprocessor
from muss.utils.helpers import count_lines, get_temp_filepath


def memoize_simplifier(simplifier):
    memo = {}

    @wraps(simplifier)
    def wrapped(complex_filepath, pred_filepath):
        complex_filehash = hashfile(complex_filepath, hexdigest=True)
        previous_pred_filepath = memo.get(complex_filehash)
        if previous_pred_filepath is not None and Path(previous_pred_filepath).exists():
            assert count_lines(complex_filepath) == count_lines(previous_pred_filepath)
            # Reuse previous prediction
            shutil.copyfile(previous_pred_filepath, pred_filepath)
        else:
            simplifier(complex_filepath, pred_filepath)
        # Save prediction
        memo[complex_filehash] = pred_filepath

    return wrapped


def make_output_file_optional(simplifier):
    @wraps(simplifier)
    def wrapped(complex_filepath, pred_filepath=None):
        if pred_filepath is None:
            pred_filepath = get_temp_filepath()
        simplifier(complex_filepath, pred_filepath)
        return pred_filepath

    return wrapped


def get_fairseq_simplifier(exp_dir, **kwargs):
    '''Function factory'''

    @make_output_file_optional
    @memoize_simplifier
    def fairseq_simplifier(complex_filepath, output_pred_filepath):
        fairseq_generate(complex_filepath, output_pred_filepath, exp_dir, **kwargs)

    return fairseq_simplifier


def get_preprocessed_simplifier(simplifier, preprocessors):
    composed_preprocessor = ComposedPreprocessor(preprocessors)

    @make_output_file_optional
    @memoize_simplifier
    @wraps(simplifier)
    def preprocessed_simplifier(complex_filepath, pred_filepath):
        preprocessed_complex_filepath = get_temp_filepath()
        composed_preprocessor.encode_file(complex_filepath, preprocessed_complex_filepath)
        preprocessed_pred_filepath = simplifier(preprocessed_complex_filepath)
        composed_preprocessor.decode_file(preprocessed_pred_filepath, pred_filepath, encoder_filepath=complex_filepath)

    preprocessed_simplifier.__name__ = f'{preprocessed_simplifier.__name__}_{composed_preprocessor.get_suffix()}'
    return preprocessed_simplifier
