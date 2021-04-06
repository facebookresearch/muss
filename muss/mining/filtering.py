# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import time

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from muss.text import get_named_entities, to_sentences
from muss.feature_extraction import get_log_ranks, get_levenshtein_distance
from muss.utils.helpers import failsafe_division


def is_contained(source, simplification):
    return source == simplification or source in simplification or simplification in source


def is_overlapping(source, simplification):
    def ordered_is_overlap(previous_text_window, next_text_window):
        '''Check if the end of one sentence is exactly the same as the beginning of the other'''
        overlapping_ratio_threshold = 0.25  # Minimum ratio of overlapping characters to be considered an overlap
        min_length = min(len(previous_text_window), len(next_text_window))
        min_overlap_length = int(min_length * overlapping_ratio_threshold) + 1
        min_possible_overlapping_text = next_text_window[:min_overlap_length]
        index = 0
        while True:
            index = previous_text_window.find(min_possible_overlapping_text, index + 1)
            if index == -1:  # Not found
                return False
            possible_overlapping_text = previous_text_window[index:]
            if next_text_window.startswith(possible_overlapping_text):
                return True

    return ordered_is_overlap(source, simplification) or ordered_is_overlap(simplification, source)


def get_quantile_log_ranks_ratio(source, simplification, q=0.9, language='en'):
    source_log_ranks = get_log_ranks(source, language=language)
    simplification_log_ranks = get_log_ranks(simplification, language=language)
    if len(source_log_ranks) == 0 or len(simplification_log_ranks) == 0:
        return 1  # Arbitrary
    return np.quantile(simplification_log_ranks, q=q) / np.quantile(source_log_ranks, q=q)


def get_max_sentence_length_ratio(source, simplification, language='en'):
    return max([len(s) for s in to_sentences(simplification, language=language)]) / max(
        [len(s) for s in to_sentences(source, language=language)]
    )


def is_different_enough(source, simplification, threshold=0.2):
    return (
        get_levenshtein_distance(source.replace(' ', '').lower(), simplification.replace(' ', '').lower()) > threshold
    )


class SimplicityScorer:
    def __init__(self, language='en'):
        self.language = language

    @property
    def feature_extractors(self):
        return {
            # We need the scores to be higher = better for the harmonic mean to work
            'log_ranks_quantile_ratio': lambda source, simplification: -get_quantile_log_ranks_ratio(
                source, simplification, language=self.language
            ),
            'max_sentence_length_ratio': lambda source, simplification: -get_max_sentence_length_ratio(
                source, simplification, language=self.language
            ),
        }

    def fit(self, pairs):
        # dir_path = VARIOUS_DIR / 'simplicity_scorers'
        # dir_path.mkdir(exist_ok=True, parents=True)
        # pairs_hash = get_string_hash(str(pairs))
        # dump_path = dir_path / pairs_hash
        n_samples = 10000
        assert len(pairs) >= n_samples
        indexes = np.random.permutation(len(pairs))[:n_samples]
        pairs = [pairs[idx] for idx in indexes]
        self.quantile_transformers = {}
        for feature_name, feature_extractor in tqdm(self.feature_extractors.items(), desc='Fitting'):
            self.quantile_transformers[feature_name] = QuantileTransformer().fit(
                np.array([feature_extractor(*pair) for pair in tqdm(pairs, desc=feature_name)]).reshape(-1, 1)
            )

    def score(self, source, simplification):
        features = []
        for feature_name, feature_extractor in self.feature_extractors.items():
            feature = self.quantile_transformers[feature_name].transform([[feature_extractor(source, simplification)]])[
                0, 0
            ]
            features.append(feature)
        return np.mean(features)


def has_hallucinated_named_entities(source, simplification, language='en'):
    def get_named_entities_label_and_text(text, language):
        named_entities = get_named_entities(text, language=language)  # format: [(label, (start, end)), ...]
        return [(label, text[start:end]) for label, (start, end) in named_entities]

    source_named_entities = get_named_entities_label_and_text(source, language)
    simplification_named_entities = get_named_entities_label_and_text(simplification, language)
    hallucinated_named_entities = Counter(simplification_named_entities) - Counter(source_named_entities)
    return len(hallucinated_named_entities) > 0


def apply_filter(pairs, filter_name, filter_function):
    start_length = len(pairs)
    start_time = time.time()
    pairs = filter_function(pairs)
    end_time = time.time()
    end_length = len(pairs)
    print(
        f'{filter_name}: reduction={failsafe_division(end_length, start_length):.2f}, {start_length} ==> {end_length}, {end_time - start_time:.1f}s'
    )
    return pairs


def filter_candidate_pairs(candidate_pairs, filters):
    n_initial_pairs = len(candidate_pairs)
    for filter_name, filter_function in filters.items():
        if filter_name.startswith('macro-'):
            filter_function_macro = filter_function
        else:
            # Transform function on individual pairs to function on all pairs
            filter_function_macro = lambda pairs: list(
                filter(filter_function, tqdm(pairs, desc=filter_name, total=len(pairs)))
            )  # noqa: E731
        candidate_pairs = apply_filter(candidate_pairs, filter_name, filter_function_macro)
        print(f'cumulated_reduction={failsafe_division(len(candidate_pairs), n_initial_pairs):.4f}')
    return candidate_pairs
