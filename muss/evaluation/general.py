# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from easse.cli import report, get_orig_and_refs_sents, evaluate_system_output

from muss.utils.helpers import write_lines, get_temp_filepath

'''A simplifier is a function with signature: simplifier(complex_filepath, output_pred_filepath)'''


def evaluate_simplifier(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None, quality_estimation=False):
    orig_sents, _ = get_orig_and_refs_sents(
        test_set, orig_sents_path=orig_sents_path, refs_sents_paths=refs_sents_paths
    )
    orig_sents_path = get_temp_filepath()
    write_lines(orig_sents, orig_sents_path)
    sys_sents_path = simplifier(orig_sents_path)
    return evaluate_system_output(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=['sari', 'bleu', 'fkgl'],
        quality_estimation=quality_estimation,
    )


def get_easse_report(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None):
    orig_sents, _ = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)
    orig_sents_path = get_temp_filepath()
    write_lines(orig_sents, orig_sents_path)
    sys_sents_path = simplifier(orig_sents_path)
    report_path = get_temp_filepath()
    report(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        report_path=report_path,
    )
    return report_path
