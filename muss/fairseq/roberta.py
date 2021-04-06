# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import re
import shutil

from muss.resources.paths import get_data_filepath, get_dataset_dir, TENSORBOARD_LOGS_DIR
from muss.utils.helpers import run_command, lock_directory, create_directory_or_skip
from muss.fairseq.base import get_fairseq_exp_dir
from muss.utils.submitit import make_checkpointable


def mlm_fairseq_preprocess(dataset):
    '''Too specific for ts.fairseq.base.fairseq_preprocess'''
    dataset_dir = get_dataset_dir(dataset)
    with lock_directory(dataset_dir):
        preprocessed_dir = dataset_dir / 'fairseq_preprocessed'
        with create_directory_or_skip(preprocessed_dir):
            vocab_path = get_data_filepath(dataset, 'vocab', 'fr')
            assert vocab_path.exists()
            trainpref = get_data_filepath(dataset, 'train', 'fr')
            validpref = get_data_filepath(dataset, 'valid', 'fr')
            testpref = get_data_filepath(dataset, 'test', 'fr')
            command = f'fairseq-preprocess --only-source --trainpref {trainpref} --validpref {validpref} --testpref {testpref} --destdir {preprocessed_dir} --workers 64 --srcdict {vocab_path}'  # noqa
            print(command)
            run_command(command)
    return preprocessed_dir


def train_roberta(
    dataset,
    sample_break_mode='complete',
    batch_size=8192,
    max_sentences=16,
    max_tokens=12000,
    tokens_per_sample=512,
    checkpoints_dir=None,
    distributed_world_size=None,
    sentencepiece_model_path=None,
    arch='roberta_base',
    dropout=0.1,
    total_updates=500000,
    log_interval=100,
    peak_lr=0.0007,
    clip_norm=None,
    no_epoch_checkpoint=False,
    validate_interval=1,
    save_interval=1,
    save_interval_updates=5000,
    warmup_updates=10000,
):
    preprocessed_dir = mlm_fairseq_preprocess(dataset)
    if checkpoints_dir is None:
        checkpoints_dir = get_fairseq_exp_dir() / 'checkpoints'
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(get_dataset_dir(dataset) / 'sentencepiece.bpe.model', checkpoints_dir)
    shutil.copy(get_dataset_dir(dataset) / 'fairseq_preprocessed/dict.txt', checkpoints_dir)
    effective_batch_size = max_sentences * distributed_world_size
    # assert batch_size % effective_batch_size == 0
    update_freq = int(round(batch_size / effective_batch_size, 0))
    print(f'batch_size={effective_batch_size * update_freq}')
    command = f'''
    eval "$(conda shell.bash hook)"  # Needed to use conda activate in subshells
    conda activate bert_fr

	fairseq-train  {preprocessed_dir} \
        --save-dir {checkpoints_dir} \
        --task masked_lm --criterion masked_lm \
        --arch {arch} --sample-break-mode {sample_break_mode} --tokens-per-sample {tokens_per_sample} \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr {peak_lr} --warmup-updates {warmup_updates} --total-num-update {total_updates} \
        --dropout {dropout} --attention-dropout {dropout} --weight-decay 0.01 \
        --max-sentences {max_sentences} --update-freq {update_freq} \
        --max-update {total_updates} --log-format simple --log-interval {log_interval} \
        --skip-invalid-size-inputs-valid-test \
        --save-interval-updates {save_interval_updates} \
        --validate-interval {validate_interval} --save-interval {save_interval} \
        --tensorboard-logdir {TENSORBOARD_LOGS_DIR} --fast-stat-sync \
        --fp16 --seed 1
    '''  # noqa
    command = command.strip(' ').strip('\n')
    if distributed_world_size is not None:
        command += f' --distributed-world-size {distributed_world_size} --distributed-port 53005'  # noqa
    if sentencepiece_model_path is not None:
        command += f' --bpe sentencepiece --sentencepiece-vocab {sentencepiece_model_path} --mask-whole-words'
    if max_tokens is not None:
        command += f' --max-tokens {max_tokens}'
    if clip_norm is not None:
        command += f' --clip-norm {clip_norm}'
    if no_epoch_checkpoint:
        command += ' --no-epoch-checkpoints'
    command = re.sub(' +', ' ', command)  # Remove multiple whitespaces
    print(command)
    run_command(command)


train_roberta = make_checkpointable(train_roberta)
