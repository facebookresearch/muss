# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_bart_kwargs, get_score_rows
from muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset
from muss.resources.datasets import create_smaller_dataset
import pandas as pd
import numpy as np
import scipy

# This dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
#prepare_wikilarge_detokenized()
#prepare_asset()

dataset = 'uts_pt_query-83c433aa147dd76db3418c194e5f47ef_db-83c433aa147dd76db3418c194e5f47ef_topk-8_nprobe-16_density-0.6_distance-0.05_filter_ne-False_levenshtein-0.2_simplicity-0.0'
kwargs = get_bart_kwargs(dataset=dataset, language='pt', use_access=True, bart_model='bart.large')
kwargs['train_kwargs']['ngpus'] = 1
kwargs['train_kwargs']['memory_efficient_fp16'] = True
kwargs['train_kwargs']['max_sentences'] = 32
kwargs['train_kwargs']['max_tokens'] = 1024
#kwargs['train_kwargs']['update_freq'] = 100
#kwargs['train_kwargs']['batch_size'] = 16
result = fairseq_train_and_evaluate_with_parametrization(**kwargs)
