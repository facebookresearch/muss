# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss

from muss.mining.preprocessing import create_base_index, get_index_name, get_sentences_paths
from muss.utils.helpers import yield_lines
from muss.laser import get_laser_embeddings
from muss.resources.paths import get_dataset_dir


# Create index
language = 'en'
n_train_sentences = 1000000
train_sentences = []
for sentences_path in get_sentences_paths(language='en'):
    for sentence in yield_lines(sentences_path):
        train_sentences.append(sentence)
        if len(train_sentences) == n_train_sentences:
            break
    if len(train_sentences) == n_train_sentences:
        break

get_embeddings = lambda sentences: get_laser_embeddings(sentences, max_tokens=3000, language=language)  # noqa: E731
output_dir = get_dataset_dir('uts') / f'base_indexes/laser_{language}'
output_dir.mkdir(exist_ok=True)
create_base_index(train_sentences, get_index_name(), get_embeddings, faiss.METRIC_INNER_PRODUCT, output_dir)
