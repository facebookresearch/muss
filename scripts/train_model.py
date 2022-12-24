# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_bart_kwargs, get_score_rows, get_mbart_kwargs
from muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset
from muss.resources.datasets import create_smaller_dataset
import pandas as pd
import numpy as np
import scipy
import argparse

# This dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
# prepare_wikilarge_detokenized()
# prepare_asset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train muss model')
    parser.add_argument('datasetname', type=str, help='dataset name')
    parser.add_argument('--language', type=str, help='language target')
    args = parser.parse_args()
    dataset = args.datasetname
    kwargs = get_mbart_kwargs(dataset=dataset, language=args.language, restore_file_path=None, use_access=True)
    kwargs['train_kwargs']['ngpus'] = 1
    kwargs['train_kwargs']['memory_efficient_fp16'] = True
    kwargs['train_kwargs']['max_sentences'] = 32
    kwargs['train_kwargs']['max_tokens'] = 1024
    # kwargs['train_kwargs']['no_epoch_checkpoints'] = True
    # kwargs['train_kwargs']['stop_min_lr'] = 0.5
    # kwargs['train_kwargs']['update_freq'] = 100
    # kwargs['train_kwargs']['batch_size'] = 16
    result = fairseq_train_and_evaluate_with_parametrization(**kwargs)
