# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_bart_kwargs, get_score_rows
from muss.resources.prepare import prepare_wikilarge_detokenized, prepare_asset
from muss.resources.datasets import create_smaller_dataset


# This dataset should exist in resources/datasets/ and contain the following files:
# train.complex, train.simple, valid.complex, valid.simple, test.complex, test.simple
prepare_wikilarge_detokenized()
prepare_asset()
dataset = 'wikilarge_detokenized'
kwargs = get_bart_kwargs(dataset=dataset, language='en', use_access=True)
kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 512  # Lower this number to prevent OOM
result = fairseq_train_and_evaluate_with_parametrization(**kwargs)
