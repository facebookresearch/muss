# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import shutil

from cachier import cachier
from easse.cli import evaluate_system_output
from easse.utils.constants import TEST_SETS_PATHS
import torch
from tqdm import tqdm

from muss.resources.paths import get_data_filepath, MODELS_DIR, get_dataset_dir
from muss.utils.helpers import add_dicts, args_str_to_dict
from muss.utils.resources import download_and_extract
from muss.preprocessors import GPT2BPEPreprocessor
from muss.preprocessing import apply_line_function_to_file
from muss.fairseq.main import get_language_from_dataset
from muss.text import truncate


def prepare_bart_model(model_name):
    bart_dir = MODELS_DIR / model_name
    if not bart_dir.exists():
        url = f'https://dl.fbaipublicfiles.com/fairseq/models/{model_name}.tar.gz'
        shutil.move(download_and_extract(url)[0], bart_dir)
        if model_name == 'bart.base':
            fix_bart_base_model_embeddings_shape()
    return bart_dir


def fix_bart_base_model_embeddings_shape():
    model_path = MODELS_DIR / 'bart.base/model.pt'
    model = torch.load(model_path)
    for key in ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']:
        model['model'][key] = model['model'][key][:50264, :]
    backup_path = model_path.parent / f'{model_path.name}.original'
    assert not backup_path.exists()
    shutil.move(model_path, backup_path)
    with open(model_path, 'wb') as f:
        torch.save(model, f)


def prepare_mbart_model():
    mbart_dir = MODELS_DIR / 'mbart'
    if not mbart_dir.exists():
        url = 'https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz'
        shutil.move(download_and_extract(url)[0], mbart_dir)
    return mbart_dir


def get_access_preprocessors_kwargs(language, use_short_name=False):
    return {
        'LengthRatioPreprocessor': {'target_ratio': 0.8, 'use_short_name': use_short_name},
        'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': 0.8, 'use_short_name': use_short_name},
        'WordRankRatioPreprocessor': {'target_ratio': 0.8, 'language': language, 'use_short_name': use_short_name},
        'DependencyTreeDepthRatioPreprocessor': {
            'target_ratio': 0.8,
            'language': language,
            'use_short_name': use_short_name,
        },
    }


def get_predict_files(language):
    return {
        'en': [get_data_filepath('asset', 'valid', 'complex'), get_data_filepath('asset', 'test', 'complex')],
        'fr': [get_data_filepath('alector', 'valid', 'complex'), get_data_filepath('alector', 'test', 'complex')],
        'es': [
            get_data_filepath('simplext_corpus', 'valid', 'complex'),
            get_data_filepath('simplext_corpus', 'test', 'complex'),
        ],
    }[language]


def get_evaluate_kwargs(language, phase='valid'):
    return {
        ('en', 'valid'): {'test_set': 'asset_valid'},
        ('en', 'test'): {'test_set': 'asset_test'},
        ('fr', 'valid'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('alector', 'valid', 'complex'),
            'refs_sents_paths': [get_data_filepath('alector', 'valid', 'simple')],
        },
        ('fr', 'test'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('alector', 'test', 'complex'),
            'refs_sents_paths': [get_data_filepath('alector', 'test', 'simple')],
        },
        ('es', 'valid'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('simplext_corpus', 'valid', 'complex'),
            'refs_sents_paths': [get_data_filepath('simplext_corpus', 'valid', 'simple')],
        },
        ('es', 'test'): {
            'test_set': 'custom',
            'orig_sents_path': get_data_filepath('simplext_corpus', 'test', 'complex'),
            'refs_sents_paths': [get_data_filepath('simplext_corpus', 'test', 'simple')],
        },
    }[(language, phase)]


def get_transformer_kwargs(dataset, language, use_access, use_short_name=False):
    kwargs = {
        'dataset': dataset,
        'parametrization_budget': 128,
        'predict_files': get_predict_files(language),
        'train_kwargs': {
            'ngpus': 8,
            'arch': 'bart_large',
            'max_tokens': 4096,
            'truncate_source': True,
            'layernorm_embedding': True,
            'share_all_embeddings': True,
            'share_decoder_input_output_embed': True,
            'required_batch_size_multiple': 1,
            'criterion': 'label_smoothed_cross_entropy',
            'lr': 3e-04,
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'weight_decay': 0.01,
            'optimizer': 'adam',
            'adam_betas': '(0.9, 0.999)',
            'adam_eps': 1e-08,
            'clip_norm': 0.1,
        },
        'preprocessors_kwargs': {
            'SentencePiecePreprocessor': {
                'vocab_size': 32000,
                'input_filepaths': [
                    get_data_filepath(dataset, 'train', 'complex'),
                    get_data_filepath(dataset, 'train', 'simple'),
                ],
            }
            # 'SentencePiecePreprocessor': {'vocab_size': 32000, 'input_filepaths':  [get_dataset_dir('enwiki') / 'all_sentences']}
        },
        'evaluate_kwargs': get_evaluate_kwargs(language),
    }
    if use_access:
        kwargs['preprocessors_kwargs'] = add_dicts(
            get_access_preprocessors_kwargs(language, use_short_name=use_short_name), kwargs['preprocessors_kwargs']
        )
    return kwargs


def get_bart_kwargs(dataset, language, use_access, use_short_name=False, bart_model='bart.large'):
    assert language == 'en'
    bart_path = prepare_bart_model(bart_model) / 'model.pt'
    arch = {
        'bart.base': 'bart_base',
        'bart.large': 'bart_large',
        'bart.large.cnn': 'bart_large',
    }[bart_model]
    kwargs = {
        'dataset': dataset,
        'metrics_coefs': [0, 1, 0],
        'parametrization_budget': 128,
        'predict_files': get_predict_files(language),
        'preprocessors_kwargs': {
            'GPT2BPEPreprocessor': {},
        },
        'preprocess_kwargs': {'dict_path': GPT2BPEPreprocessor().dict_path},
        'train_kwargs': {
            'ngpus': 8,
            'arch': arch,
            'restore_file': bart_path,
            'max_tokens': 4096,
            'lr': 3e-05,
            'warmup_updates': 500,
            'truncate_source': True,
            'layernorm_embedding': True,
            'share_all_embeddings': True,
            'share_decoder_input_output_embed': True,
            'reset_optimizer': True,
            'reset_dataloader': True,
            'reset_meters': True,
            'required_batch_size_multiple': 1,
            'criterion': 'label_smoothed_cross_entropy',
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'weight_decay': 0.01,
            'optimizer': 'adam',
            'adam_betas': '(0.9, 0.999)',
            'adam_eps': 1e-08,
            'clip_norm': 0.1,
            'lr_scheduler': 'polynomial_decay',
            'max_update': 20000,
            'skip_invalid_size_inputs_valid_test': True,
            'find_unused_parameters': True,
        },
        'evaluate_kwargs': get_evaluate_kwargs(language),
    }
    if use_access:
        kwargs['preprocessors_kwargs'] = add_dicts(
            get_access_preprocessors_kwargs(language, use_short_name=use_short_name), kwargs['preprocessors_kwargs']
        )
    return kwargs


def get_mbart_kwargs(dataset, language, use_access, use_short_name=False):
    mbart_dir = prepare_mbart_model()
    mbart_path = mbart_dir / 'model.pt'
    # source_lang = f'{language}_XX'
    # target_lang = f'{language}_XX'
    source_lang = 'complex'
    target_lang = 'simple'
    kwargs = {
        'dataset': dataset,
        'metrics_coefs': [0, 1, 0],
        'parametrization_budget': 128,
        'predict_files': get_predict_files(language),
        'preprocessors_kwargs': {
            'SentencePiecePreprocessor': {
                'sentencepiece_model_path': mbart_dir / 'sentence.bpe.model',
                'tokenize_special_tokens': True,
            },
        },
        'preprocess_kwargs': {
            'dict_path': mbart_dir / 'dict.txt',
            'source_lang': source_lang,
            'target_lang': target_lang,
        },
        'train_kwargs': add_dicts(
            {'ngpus': 8},
            args_str_to_dict(
                f'''--restore-file {mbart_path}  --arch mbart_large --task translation_from_pretrained_bart  --source-lang {source_lang} --target-lang {target_lang}  --encoder-normalize-before --decoder-normalize-before --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 1024 --update-freq 2 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
     --layernorm-embedding  --ddp-backend no_c10d'''
            ),
        ),  # noqa: E501
        'generate_kwargs': args_str_to_dict(
            f'''--task translation_from_pretrained_bart --source_lang {source_lang} --target-lang {target_lang} --batch-size 32 --langs ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN'''  # noqa: E501
        ),
        'evaluate_kwargs': get_evaluate_kwargs(language),
    }
    if use_access:
        kwargs['preprocessors_kwargs'] = add_dicts(
            get_access_preprocessors_kwargs(language, use_short_name=use_short_name), kwargs['preprocessors_kwargs']
        )
    return kwargs


def get_all_baseline_rows():
    paths = {
        ('asset', 'test'): ('en', TEST_SETS_PATHS[('asset_test', 'orig')], TEST_SETS_PATHS[('asset_test', 'refs')]),
        ('asset', 'valid'): ('en', TEST_SETS_PATHS[('asset_valid', 'orig')], TEST_SETS_PATHS[('asset_valid', 'refs')]),
        ('turkcorpus_detokenized', 'test'): (
            'en',
            TEST_SETS_PATHS[('turkcorpus_test', 'orig')],
            TEST_SETS_PATHS[('turkcorpus_test', 'refs')],
        ),
        ('turkcorpus_detokenized', 'valid'): (
            'en',
            TEST_SETS_PATHS[('turkcorpus_valid', 'orig')],
            TEST_SETS_PATHS[('turkcorpus_valid', 'refs')],
        ),
        ('alector', 'test'): (
            'fr',
            get_data_filepath('alector', 'test', 'complex'),
            [get_data_filepath('alector', 'test', 'simple')],
        ),
        ('alector', 'valid'): (
            'fr',
            get_data_filepath('alector', 'valid', 'complex'),
            [get_data_filepath('alector', 'valid', 'simple')],
        ),
        # Old dataset with problems
        ('simplext_corpus_all', 'test'): (
            'es',
            get_data_filepath('simplext_corpus_all', 'test', 'complex'),
            [get_data_filepath('simplext_corpus_all', 'test', 'simple')],
        ),
        ('simplext_corpus_all', 'valid'): (
            'es',
            get_data_filepath('simplext_corpus_all', 'valid', 'complex'),
            [get_data_filepath('simplext_corpus_all', 'valid', 'simple')],
        ),
        ('simplext_corpus_all_fixed', 'test'): (
            'es',
            get_data_filepath('simplext_corpus_all_fixed', 'test', 'complex'),
            [get_data_filepath('simplext_corpus_all_fixed', 'test', 'simple')],
        ),
        ('simplext_corpus_all_fixed', 'valid'): (
            'es',
            get_data_filepath('simplext_corpus_all_fixed', 'valid', 'complex'),
            [get_data_filepath('simplext_corpus_all_fixed', 'valid', 'simple')],
        ),
        ('simpitiki', 'test'): (
            'it',
            get_data_filepath('simpitiki', 'test', 'complex'),
            [get_data_filepath('simpitiki', 'test', 'simple')],
        ),
        ('simpitiki', 'valid'): (
            'it',
            get_data_filepath('simpitiki', 'valid', 'complex'),
            [get_data_filepath('simpitiki', 'valid', 'simple')],
        ),
    }
    rows = []
    for (dataset, phase), (language, orig_sents_path, refs_sents_paths) in tqdm(paths.items()):
        dataset_rows = get_baseline_rows(orig_sents_path, tuple(refs_sents_paths), language)
        for row in dataset_rows:
            row['dataset'] = dataset
            row['phase'] = phase
        rows.extend(dataset_rows)
    return rows


@cachier()
def get_baseline_rows(orig_sents_path, refs_sents_paths, language):
    refs_sents_paths = list(refs_sents_paths)
    rows = []
    scores = evaluate_system_output(
        'custom',
        sys_sents_path=orig_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=['sari', 'bleu', 'fkgl', 'sari_by_operation'],
        quality_estimation=False,
    )
    row = {
        'exp_name': 'Identity',
        'language': language,
    }
    rows.append(add_dicts(row, scores))

    scores = evaluate_system_output(
        'custom',
        sys_sents_path=apply_line_function_to_file(
            lambda sentence: truncate(sentence, truncate_prop=0.2, language=language), orig_sents_path
        ),
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=['sari', 'bleu', 'fkgl', 'sari_by_operation'],
        quality_estimation=False,
    )
    row = {
        'exp_name': 'Truncate',
        'language': language,
    }
    rows.append(add_dicts(row, scores))

    if len(refs_sents_paths) > 1:
        for i in range(len(refs_sents_paths)):
            scores = evaluate_system_output(
                'custom',
                sys_sents_path=refs_sents_paths[i],
                orig_sents_path=orig_sents_path,
                refs_sents_paths=[refs_sents_paths[i - 1]] + refs_sents_paths[:i] + refs_sents_paths[i + 1 :],
                metrics=['sari', 'bleu', 'fkgl', 'sari_by_operation'],
                quality_estimation=False,
            )
            row = {
                'exp_name': 'Reference',
                'language': language,
                'job_id': f'ref_{i}',
            }
            rows.append(add_dicts(row, scores))
    return rows


@cachier()
def get_scores_on_dataset(pred_path, dataset, phase):
    orig_sents_path = get_data_filepath(dataset, phase, 'complex')
    refs_sents_paths = list(get_dataset_dir(dataset).glob(f'{phase}.simple*'))
    return evaluate_system_output(
        'custom',
        sys_sents_path=pred_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=['sari', 'bleu', 'fkgl', 'sari_by_operation'],
        quality_estimation=False,
    )


def get_score_rows(exp_dir, kwargs, additional_fields=None):
    rows = []
    language = get_language_from_dataset(kwargs['dataset'])
    for pred_path in exp_dir.glob('finetune_*.pred'):
        dataset, phase = re.match(r'finetune_.+?_valid-test_(.+)_(.+?).pred', pred_path.name).groups()
        scores = get_scores_on_dataset(pred_path, dataset, phase)
        row = {
            'language': language,
            'dataset': dataset,
            'phase': phase,
        }
        if additional_fields is not None:
            row = add_dicts(row, additional_fields)
        rows.append(add_dicts(row, scores))
    return rows
