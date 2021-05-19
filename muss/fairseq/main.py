# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import shutil
import warnings

import nevergrad as ng
import numpy as np
from submitit.helpers import DelayedSubmission

from muss.evaluation.general import evaluate_simplifier, get_easse_report, get_orig_and_refs_sents
from muss.evaluation.utils import combine_metrics
from muss.fairseq.base import fairseq_preprocess, fairseq_train, get_fairseq_exp_dir
from muss.resources.datasets import has_lines_in_common
from muss.preprocessors import get_preprocessors, get_preprocessor_by_name
from muss.resources.datasets import create_preprocessed_dataset
from muss.resources.paths import get_data_filepath, get_dataset_dir
from muss.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from muss.utils.submitit import get_job_id
from muss.utils.helpers import print_running_time, add_dicts


def check_dataset(dataset):
    # Sanity check with evaluation dataset
    if has_lines_in_common(
        get_data_filepath(dataset, 'train', 'complex'), get_data_filepath('asset', 'valid', 'complex')
    ):
        warnings.warn('WARNING: Dataset has validation samples in training set!')
    if has_lines_in_common(
        get_data_filepath(dataset, 'train', 'complex'), get_data_filepath('asset', 'test', 'complex')
    ):
        warnings.warn('WARNING: Dataset has test samples in training set!')


def prepare_exp_dir():
    exp_dir = get_fairseq_exp_dir()
    exp_dir.mkdir(exist_ok=True, parents=True)
    print(f'exp_dir={exp_dir}')
    return exp_dir


def check_and_resolve_args(kwargs):
    if kwargs.get('diverse_beam_groups_ratio', None) is not None:
        diverse_beam_groups = max(int(kwargs['beam'] * kwargs['diverse_beam_groups_ratio']), 1)
        print(f'diverse_beam_groups={diverse_beam_groups}')
        assert kwargs['beam'] % diverse_beam_groups == 0
        kwargs['diverse_beam_groups'] = diverse_beam_groups
    else:
        diverse_beam_groups = None
    return kwargs


def fairseq_prepare_and_train(dataset, **kwargs):
    check_dataset(dataset)
    kwargs = check_and_resolve_args(kwargs)
    exp_dir = prepare_exp_dir()
    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    preprocessors = get_preprocessors(preprocessors_kwargs)
    if len(preprocessors) > 0:
        dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=8)
        dataset_dir = get_dataset_dir(dataset)
        shutil.copy(dataset_dir / 'preprocessors.pickle', exp_dir)
        if hasattr(preprocessors[-1], 'copy_sentencepiece_files_to_dir'):
            preprocessors[-1].copy_sentencepiece_files_to_dir(dataset_dir)
    model_symlink_path = exp_dir / 'model.pt'
    if not model_symlink_path.exists():
        model_symlink_path.symlink_to('checkpoints/checkpoint_best.pt')
    preprocessed_dir = fairseq_preprocess(dataset, **kwargs.get('preprocess_kwargs', {}))
    train_kwargs = kwargs.get('train_kwargs', {})
    fairseq_train(preprocessed_dir, exp_dir=exp_dir, **train_kwargs)
    return exp_dir


def fairseq_get_simplifier(exp_dir, **kwargs):
    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    generate_kwargs = kwargs.get('generate_kwargs', {})
    preprocessors = get_preprocessors(preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(exp_dir, **generate_kwargs)
    return get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)


def get_predictions(source_path, exp_dir, **kwargs):
    simplifier = fairseq_get_simplifier(exp_dir, **kwargs)
    return simplifier(source_path)


def fairseq_evaluate(exp_dir, **kwargs):
    simplifier = fairseq_get_simplifier(exp_dir, **kwargs)
    evaluate_kwargs = kwargs.get('evaluate_kwargs', {'test_set': 'asset_valid'})
    return evaluate_simplifier(simplifier, **evaluate_kwargs)


def get_easse_report_from_exp_dir(exp_dir, **kwargs):
    simplifier = fairseq_get_simplifier(exp_dir, **kwargs)
    return get_easse_report(simplifier, **kwargs.get('evaluate_kwargs', {'test_set': 'asset_valid'}))


def fairseq_evaluate_and_save(exp_dir, **kwargs):
    scores = fairseq_evaluate(exp_dir, **kwargs)
    print(f'scores={scores}')
    report_path = exp_dir / 'easse_report.html'
    shutil.move(get_easse_report_from_exp_dir(exp_dir, **kwargs), report_path)
    print(f'report_path={report_path}')
    predict_files = kwargs.get(
        'predict_files', [get_data_filepath('asset', 'valid', 'complex'), get_data_filepath('asset', 'test', 'complex')]
    )
    for source_path in predict_files:
        pred_path = get_predictions(source_path, exp_dir, **kwargs)
        shutil.copyfile(source_path, exp_dir / source_path.name)
        new_pred_path = exp_dir / source_path.with_suffix('.pred').name
        shutil.move(pred_path, new_pred_path)
        print(f'source_path={source_path}')
        print(f'pred_path={new_pred_path}')
    return scores


def find_best_parametrization_nevergrad(
    exp_dir, preprocessors_kwargs, metrics_coefs=[0, 1, 0], parametrization_budget=64, **kwargs
):
    def evaluate_parametrization(**preprocessors_kwargs):
        simplifier = fairseq_get_simplifier(
            exp_dir, preprocessors_kwargs=preprocessors_kwargs, generate_kwargs=kwargs.get('generate_kwargs', {})
        )
        scores = evaluate_simplifier(simplifier, **kwargs.get('evaluate_kwargs', {'test_set': 'asset_valid'}))
        return combine_metrics(scores['bleu'], scores['sari'], scores['fkgl'], metrics_coefs)

    def get_parametrization(preprocessors_kwargs):
        parametrization_kwargs = {}
        for preprocessor_name, preprocessor_kwargs in preprocessors_kwargs.items():
            assert '_' not in preprocessor_name
            nevergrad_variables = add_dicts(
                preprocessor_kwargs, get_preprocessor_by_name(preprocessor_name).get_nevergrad_variables()
            )
            parametrization_kwargs[preprocessor_name] = ng.p.Dict(**nevergrad_variables)
        return ng.p.Instrumentation(**parametrization_kwargs)

    parametrization = get_parametrization(preprocessors_kwargs)
    if parametrization.dimension == 0:
        return preprocessors_kwargs
    # No need to search a lot when there are only a few parameters
    parametrization_budget = min(32 ** parametrization.dimension, parametrization_budget)
    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=parametrization_budget, num_workers=1)
    optimizer.register_callback("tell", ng.callbacks.ProgressBar())
    recommendation = optimizer.minimize(evaluate_parametrization, verbosity=0)
    return recommendation.kwargs


def find_best_parametrization_fast(exp_dir, preprocessors_kwargs, **kwargs):
    preprocessors_kwargs = preprocessors_kwargs.copy()  # We are going to modify it inplace
    preprocessors = get_preprocessors(preprocessors_kwargs)
    orig_sents, refs_sents = get_orig_and_refs_sents(**kwargs.get('evaluate_kwargs', {'test_set': 'asset_valid'}))
    features = defaultdict(list)
    for ref_sents in refs_sents:
        for orig_sent, ref_sent in zip(orig_sents, ref_sents):
            for preprocessor in preprocessors:
                if not hasattr(preprocessor, 'get_feature_value'):
                    continue
                features[preprocessor.__class__.__name__].append(preprocessor.get_feature_value(orig_sent, ref_sent))
    for preprocessor_name, preprocessor_features in features.items():
        preprocessors_kwargs[preprocessor_name]['target_ratio'] = np.mean(preprocessor_features)
    return preprocessors_kwargs


def find_best_parametrization(exp_dir, preprocessors_kwargs, fast_parametrization_search=False, *args, **kwargs):
    if fast_parametrization_search:
        return find_best_parametrization_fast(exp_dir, preprocessors_kwargs, *args, **kwargs)
    else:
        return find_best_parametrization_nevergrad(exp_dir, preprocessors_kwargs, *args, **kwargs)


def get_language_from_dataset(dataset):
    # TODO: Should be in ts.uts.training
    if '_fr_' in dataset:
        return 'fr'
    if '_es_' in dataset:
        return 'es'
    if '_it_' in dataset:
        return 'it'
    else:
        return 'en'


def get_datasets_for_language(language):
    # TODO: Should be in ts.uts.training
    return {
        'en': ['asset', 'turkcorpus_detokenized'],
        'fr': ['alector'],
        'es': ['simplext_corpus_all_fixed'],
        # 'it': ['simpitiki']
    }[language]


def finetune_and_predict_on_dataset(finetuning_dataset, exp_dir, **kwargs):
    kwargs['train_kwargs']['ngpus'] = 1
    prefix = 'finetune'
    if kwargs.get('fast_parametrization_search', False):
        prefix += '_fast'
    pred_filepaths = [
        exp_dir / f'{prefix}_{finetuning_dataset}_valid-test_{finetuning_dataset}_valid.pred',
        exp_dir / f'{prefix}_{finetuning_dataset}_valid-test_{finetuning_dataset}_test.pred',
    ]
    if all([path.exists() for path in pred_filepaths]):
        return
    for phase, pred_filepath in zip(['valid', 'test'], pred_filepaths):
        orig_sents_path = get_data_filepath(finetuning_dataset, phase, 'complex')
        refs_sents_paths = list(get_dataset_dir(finetuning_dataset).glob(f'{phase}.simple*'))
        kwargs['evaluate_kwargs'] = {
            'test_set': 'custom',
            'orig_sents_path': orig_sents_path,
            'refs_sents_paths': refs_sents_paths,
        }
        if phase == 'valid':
            # Finetune preprocessors_kwargs only on valid
            kwargs['preprocessors_kwargs'] = find_best_parametrization(exp_dir, **kwargs)
        shutil.copyfile(fairseq_get_simplifier(exp_dir, **kwargs)(orig_sents_path), pred_filepath)


def fairseq_train_and_evaluate_with_parametrization(dataset, **kwargs):
    # Training
    exp_dir = print_running_time(fairseq_prepare_and_train)(dataset, **kwargs)
    # Find best parametrization
    recommended_preprocessors_kwargs = print_running_time(find_best_parametrization)(exp_dir, **kwargs)
    print(f'recommended_preprocessors_kwargs={recommended_preprocessors_kwargs}')
    kwargs['preprocessor_kwargs'] = recommended_preprocessors_kwargs
    # Evaluation
    scores = print_running_time(fairseq_evaluate_and_save)(exp_dir, **kwargs)
    score = combine_metrics(scores['bleu'], scores['sari'], scores['fkgl'], kwargs.get('metrics_coefs', [0, 1, 0]))
    # TODO: This is a redundant hack with what happens in fairseq_evaluate_and_save (predict_files and evaluate_kwargs), it should be fixed
    language = get_language_from_dataset(dataset)
    for finetuning_dataset in get_datasets_for_language(language):
        finetune_and_predict_on_dataset(finetuning_dataset, exp_dir, **kwargs)
    return score


def checkpoint_fairseq_train_and_evaluate_with_parametrization(*args, **kwargs):
    last_checkpoint_path = get_fairseq_exp_dir(get_job_id()) / 'checkpoints/checkpoint_last.pt'
    if last_checkpoint_path.exists():
        # We don't want to restore from pretrained models again but from the last saved checkpoint
        kwargs['train_kwargs']['restore_file'] = last_checkpoint_path
    return DelayedSubmission(fairseq_train_and_evaluate_with_parametrization, *args, **kwargs)  # submits to requeuing


fairseq_train_and_evaluate_with_parametrization.checkpoint = checkpoint_fairseq_train_and_evaluate_with_parametrization
