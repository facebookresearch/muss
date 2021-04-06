# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import Levenshtein
import numpy as np

from muss.resources.paths import get_fasttext_embeddings_path
from muss.text import spacy_process, get_content_words
from muss.utils.helpers import failsafe_division, yield_lines


@lru_cache(maxsize=10)
def get_word2rank(vocab_size=10 ** 5, language='en'):
    word2rank = {}
    line_generator = yield_lines(get_fasttext_embeddings_path(language))
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank


def get_rank(word, language='en'):
    return get_word2rank(language=language).get(word, len(get_word2rank(language=language)))


def get_log_rank(word, language='en'):
    return np.log(1 + get_rank(word, language=language))


def get_log_ranks(text, language='en'):
    return [
        get_log_rank(word, language=language)
        for word in get_content_words(text, language=language)
        if word in get_word2rank(language=language)
    ]


# Single sentence feature extractors with signature function(sentence) -> float
def get_lexical_complexity_score(sentence, language='en'):
    log_ranks = get_log_ranks(sentence, language=language)
    if len(log_ranks) == 0:
        log_ranks = [np.log(1 + len(get_word2rank()))]  # TODO: This is completely arbitrary
    return np.quantile(log_ranks, 0.75)


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


def get_levenshtein_distance(complex_sentence, simple_sentence):
    # We should rename this to get_levenshtein_distance_ratio for more clarity
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)


def get_replace_only_levenshtein_distance(complex_sentence, simple_sentence):
    return len(
        [_ for operation, _, _ in Levenshtein.editops(complex_sentence, simple_sentence) if operation == 'replace']
    )


def get_replace_only_levenshtein_distance_ratio(complex_sentence, simple_sentence):
    max_replace_only_distance = min(len(complex_sentence), len(simple_sentence))
    return failsafe_division(
        get_replace_only_levenshtein_distance(complex_sentence, simple_sentence), max_replace_only_distance, default=0
    )


def get_replace_only_levenshtein_similarity(complex_sentence, simple_sentence):
    return 1 - get_replace_only_levenshtein_distance_ratio(complex_sentence, simple_sentence)


def get_dependency_tree_depth(sentence, language='en'):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [
        get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence, language=language).sents
    ]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)
