# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from pathlib import Path
import random
import re
import shutil
import shlex
import time

from fairseq_cli import preprocess, train, generate

from muss.text import remove_multiple_whitespaces
from muss.resources.paths import get_dataset_dir, EXP_DIR, LANGUAGES, get_data_filepath, PHASES
from muss.utils.submitit import get_job_id
from muss.utils.training import clear_cuda_cache
from muss.utils.helpers import (
    log_std_streams,
    lock_directory,
    create_directory_or_skip,
    yield_lines,
    write_lines,
    mock_cli_args,
    create_temp_dir,
    mute,
    args_dict_to_str,
)


def get_fairseq_exp_dir(job_id=None):
    if job_id is None:
        job_id = get_job_id()
    if job_id is not None:
        dir_name = f'slurmjob_{job_id}'
    else:
        dir_name = f'local_{int(time.time() * 1000)}'
    return Path(EXP_DIR) / f'fairseq' / dir_name


def fairseq_preprocess(dataset, dict_path=None, source_lang='complex', target_lang='simple'):
    dataset_dir = get_dataset_dir(dataset)
    with lock_directory(dataset_dir):
        preprocessed_dir = dataset_dir / f'fairseq_preprocessed_{source_lang}-{target_lang}'
        with create_directory_or_skip(preprocessed_dir):
            # HACK
            for phase in PHASES:
                for language, new_language in zip(LANGUAGES, [source_lang, target_lang]):
                    symlink_path = get_data_filepath(dataset, phase, new_language)
                    if not symlink_path.exists():
                        symlink_path.symlink_to(get_data_filepath(dataset, phase, language))
            trainpref = str(get_data_filepath(dataset, 'train', 'dummy')).replace('.dummy', '')
            validpref = str(get_data_filepath(dataset, 'valid', 'dummy')).replace('.dummy', '')
            testpref = str(get_data_filepath(dataset, 'test', 'dummy')).replace('.dummy', '')
            args = f'''
                --source-lang {source_lang} --target-lang {target_lang} --trainpref {trainpref} --validpref {validpref} --testpref {testpref}
                --destdir {preprocessed_dir} --bpe sentencepiece
                --joined-dictionary --workers 32
            '''
            if dict_path is not None:
                args = f'{args} --srcdict {dict_path}'
            args = remove_multiple_whitespaces(args.replace('\n', ' ')).strip(' ')
            print(f'fairseq-preprocess {args}')
            args = shlex.split(args)
            with mock_cli_args(args):
                preprocess.cli_main()
        return preprocessed_dir


@clear_cuda_cache
def fairseq_train(
    preprocessed_dir,
    exp_dir,
    ngpus=1,
    batch_size=8192,  # Batch size across all gpus (taking update freq into account)
    max_sentences=64,  # Max sentences per GPU
    arch='transformer',
    save_interval_updates=100,
    max_update=50000,
    lr=0.001,
    warmup_updates=4000,
    dropout=0.1,
    lr_scheduler='inverse_sqrt',
    criterion='label_smoothed_cross_entropy',
    seed=None,
    fp16=True,
    **kwargs,
):
    with log_std_streams(exp_dir / 'fairseq_train.stdout'):
        exp_dir = Path(exp_dir)
        preprocessed_dir = Path(preprocessed_dir)
        exp_dir.mkdir(exist_ok=True, parents=True)
        # Copy dictionaries to exp_dir for generation
        for dict_path in preprocessed_dir.glob('dict.*.txt'):
            shutil.copy(dict_path, exp_dir)
        checkpoints_dir = exp_dir / 'checkpoints'
        total_real_batch_size = max_sentences * ngpus
        update_freq = int(round(batch_size / total_real_batch_size, 0))
        if seed is None:
            seed = random.randint(0, 1000)
        distributed_port = random.randint(10000, 20000)
        args = f'''
        {preprocessed_dir} --task translation --source-lang complex --target-lang simple --save-dir {checkpoints_dir}
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0
        --criterion {criterion} --label-smoothing 0.1
        --lr-scheduler {lr_scheduler} --lr {lr} --warmup-updates {warmup_updates} --update-freq {update_freq}
        --arch {arch} --dropout {dropout} --weight-decay 0.0 --clip-norm 0.1 --share-all-embeddings
        --no-epoch-checkpoints --save-interval 999999 --validate-interval 999999
        --max-update {max_update} --save-interval-updates {save_interval_updates} --keep-interval-updates 1 --patience 10
        --batch-size {max_sentences} --seed {seed}
        --distributed-world-size {ngpus} --distributed-port {distributed_port}
        '''
        if lr_scheduler == 'inverse_sqrt':
            args += '--warmup-init-lr 1e-07'
        if fp16:
            args += f' --fp16'
        # FIXME: if the kwargs are already present in the args string, they will appear twice but fairseq will take only the last one into account
        args += f' {args_dict_to_str(kwargs)}'
        args = remove_multiple_whitespaces(args.replace('\n', ' ')).strip(' ')
        # Recover lost quotes around adam betas
        args = re.sub(r'--adam-betas (\(0\.\d+, 0\.\d+\))', r"--adam-betas '\1'", args)
        print(f'fairseq-train {args}')
        with mock_cli_args(shlex.split(args)):
            train.cli_main()


def fairseq_parse_all_hypotheses(out_filepath):
    hypotheses_dict = defaultdict(list)
    for line in yield_lines(out_filepath):
        match = re.match(r'^H-(\d+)\t-?\d+\.\d+\t(.*)$', line)
        if match:
            sample_id, hypothesis = match.groups()
            hypotheses_dict[int(sample_id)].append(hypothesis)
    # Sort in original order
    return [hypotheses_dict[i] for i in range(len(hypotheses_dict))]


@clear_cuda_cache
def _fairseq_generate(
    complex_filepath,
    output_pred_filepath,
    checkpoint_paths,
    complex_dictionary_path,
    simple_dictionary_path,
    beam=5,
    hypothesis_num=1,
    lenpen=1.0,
    diverse_beam_groups=None,
    diverse_beam_strength=0.5,
    sampling=False,
    max_tokens=16384,
    source_lang='complex',
    target_lang='simple',
    **kwargs,
):
    # exp_dir must contain checkpoints/checkpoint_best.pt, and dict.{complex,simple}.txt
    # First copy input complex file to exp_dir and create dummy simple file
    with create_temp_dir() as temp_dir:
        new_complex_filepath = temp_dir / f'tmp.{source_lang}-{target_lang}.{source_lang}'
        dummy_simple_filepath = temp_dir / f'tmp.{source_lang}-{target_lang}.{target_lang}'
        shutil.copy(complex_filepath, new_complex_filepath)
        shutil.copy(complex_filepath, dummy_simple_filepath)
        shutil.copy(complex_dictionary_path, temp_dir / f'dict.{source_lang}.txt')
        shutil.copy(simple_dictionary_path, temp_dir / f'dict.{target_lang}.txt')
        args = f'''
        {temp_dir} --dataset-impl raw --gen-subset tmp --path {':'.join([str(path) for path in checkpoint_paths])}
        --beam {beam} --nbest {hypothesis_num} --lenpen {lenpen}
        --diverse-beam-groups {diverse_beam_groups if diverse_beam_groups is not None else -1} --diverse-beam-strength {diverse_beam_strength}
        --max-tokens {max_tokens}
        --model-overrides "{{'encoder_embed_path': None, 'decoder_embed_path': None}}"
        --skip-invalid-size-inputs-valid-test
        '''
        if sampling:
            args += f'--sampling --sampling-topk 10'
        # FIXME: if the kwargs are already present in the args string, they will appear twice but fairseq will take only the last one into account
        args += f' {args_dict_to_str(kwargs)}'
        args = remove_multiple_whitespaces(args.replace('\n', ' '))
        out_filepath = temp_dir / 'generation.out'
        with mute(mute_stderr=False):
            with log_std_streams(out_filepath):
                # evaluate model in batch mode
                print(f'fairseq-generate {args}')
                args = shlex.split(args)
                with mock_cli_args(args):
                    generate.cli_main()

        all_hypotheses = fairseq_parse_all_hypotheses(out_filepath)
        predictions = [hypotheses[hypothesis_num - 1] for hypotheses in all_hypotheses]
        write_lines(predictions, output_pred_filepath)


def fairseq_generate(
    complex_filepath,
    output_pred_filepath,
    exp_dir,
    beam=5,
    hypothesis_num=1,
    lenpen=1.0,
    diverse_beam_groups=None,
    diverse_beam_strength=0.5,
    sampling=False,
    max_tokens=8000,
    source_lang='complex',
    target_lang='simple',
    **kwargs,
):
    exp_dir = Path(exp_dir)
    possible_checkpoint_paths = [
        exp_dir / 'model.pt',
        exp_dir / 'checkpoints/checkpoint_best.pt',
        exp_dir / 'checkpoints/checkpoint_last.pt',
    ]
    assert any(
        [path for path in possible_checkpoint_paths if path.exists()]
    ), f'Generation failed, no checkpoint found in {possible_checkpoint_paths}'  # noqa: E501
    checkpoint_path = [path for path in possible_checkpoint_paths if path.exists()][0]
    complex_dictionary_path = exp_dir / f'dict.{source_lang}.txt'
    simple_dictionary_path = exp_dir / f'dict.{target_lang}.txt'
    _fairseq_generate(
        complex_filepath,
        output_pred_filepath,
        [checkpoint_path],
        complex_dictionary_path=complex_dictionary_path,
        simple_dictionary_path=simple_dictionary_path,
        beam=beam,
        hypothesis_num=hypothesis_num,
        lenpen=lenpen,
        diverse_beam_groups=diverse_beam_groups,
        diverse_beam_strength=diverse_beam_strength,
        sampling=sampling,
        max_tokens=max_tokens,
        **kwargs,
    )
