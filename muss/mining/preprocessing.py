# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import gzip
from pathlib import Path
from string import punctuation

from tqdm import tqdm
import numpy as np
import faiss

from muss.preprocessing import normalize_punctuation
from muss.text import yield_sentence_concatenations, normalize_unicode
from muss.kenlm import get_kenlm_log_prob
from muss.utils.helpers import batch_items, log_action, yield_lines
from muss.resources.paths import RESOURCES_DIR
from muss.mining.nn_search import cached_count_lines


def yield_json_documents_from_compressed(compressed_path):
    for document in yield_lines(compressed_path, gzipped=True):
        yield json.loads(document)


def split_ccnet_shard(shard_path, output_dir, n_docs_per_subshard=10000):
    '''We need to split the shards even more for the embeddings to fit in memory'''

    def write_lines_to_compressed_file(lines, compressed_filepath):
        with gzip.open(compressed_filepath, 'wt', compresslevel=1) as f:
            for line in lines:
                if not line.endswith('\n'):
                    line = line + '\n'
                f.write(line)

    if output_dir.exists():
        return
    assert str(shard_path).endswith('.json.gz')
    shard_path = Path(shard_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    with gzip.open(shard_path, 'rt') as f:
        for file_number, lines in enumerate(batch_items(item_generator=f, batch_size=n_docs_per_subshard)):
            assert file_number < 1000
            output_filepath = Path(output_dir) / f'{file_number:03d}.json.gz'
            if not output_filepath.exists():
                write_lines_to_compressed_file(lines, output_filepath)


def has_too_much_punctuation(text):
    characters = text.replace(' ', '')
    punctuation_count = len([c for c in characters if c in punctuation])
    return punctuation_count / len(characters) > 0.1


def has_low_lm_prob(text, language):
    # The slope is the linear coefficient that links the log probability and the length of the sentence in characters
    model_dir, slope = {
        'en': (RESOURCES_DIR / 'models/language_models/kenlm_enwiki', -0.6),
        'fr': (RESOURCES_DIR / 'models/language_models/kenlm_frwiki', -0.6),
        'es': (RESOURCES_DIR / 'models/language_models/kenlm_ccnet_es', -0.8),
        'it': (RESOURCES_DIR / 'models/language_models/kenlm_ccnet_it', -0.8),
    }[language]
    if not model_dir.exists():
        print(
            f'WARNING: no kenlm language model found for {language}, you need to train your own (see https://github.com/kpu/kenlm). Skipping language model filtering.'  # noqa: E501
        )
        return False
    return get_kenlm_log_prob(text, model_dir) / len(text) < slope


def sentence_tokenize_document(document, language):
    document = document.replace('\n', ' ').replace('\x00', ' ').replace('\t', ' ')
    document = normalize_punctuation(normalize_unicode(document))
    sentences = list(yield_sentence_concatenations(document, min_length=10, max_length=300, language=language))
    # Filter out sentences (too short, too much punctuation, low lm prob)
    sentences = list(filter(lambda sentence: len(sentence) >= 30, sentences))
    sentences = list(filter(lambda sentence: not has_too_much_punctuation(sentence), sentences))
    sentences = list(filter(lambda sentence: not has_low_lm_prob(sentence, language), sentences))
    return sentences


def sentence_tokenize_subshard(subshard_path, sentences_path, language):
    if not sentences_path.exists():
        with log_action('Sentence tokenization'):
            with gzip.open(sentences_path, 'wt', compresslevel=1) as f:
                for json_document in tqdm(yield_json_documents_from_compressed(subshard_path), desc='documents'):
                    sentences = sentence_tokenize_document(json_document.pop('raw_content'), language=language)
                    for sentence in sentences:
                        f.write(f'{sentence}\n')
        cached_count_lines(sentences_path)  # Cache line count
    return sentences_path


def get_subshard_paths(raw_original_dir):
    return list(sorted(raw_original_dir.glob('**/*.json.gz')))


def get_sentences_paths(dataset_dir):
    return list(sorted((dataset_dir / 'sentences').glob('*.txt.gz')))


def get_n_cells(n_total_samples):
    return int(2 ** round(np.log2(np.sqrt(n_total_samples))))


def get_index_name():
    n_total_samples = int(1e9)
    n_cells = get_n_cells(n_total_samples)
    sq_size = 8
    pca_dim = 512
    embeddings_dim = 1024
    index_size = (n_total_samples * embeddings_dim * 4) / (32 / sq_size) / (embeddings_dim / pca_dim)
    print(f'Expected index size: {index_size // 1e9}GB')
    return f'PCAR{pca_dim},IVF{n_cells},SQ{sq_size}'


def create_base_index(sentences, index_name, get_embeddings, metric, output_dir):
    index_prefix = f'{index_name.replace(",", "_").lower()}_metric{metric}'
    index_path = output_dir / f'{index_prefix}.faiss_index'
    if not index_path.exists():
        with log_action('Computing embeddings'):
            embeddings = get_embeddings(sentences)
        with log_action('Training index'):
            index = faiss.index_factory(embeddings.shape[1], index_name, metric)
            index.train(embeddings)
        faiss.write_index(index, str(index_path))
    return index_path
