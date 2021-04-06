# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from muss.resources.paths import LASER_DIR
from muss.preprocessing import get_parallel_file_preprocessor
from muss.utils.training import clear_cuda_cache
from muss.utils.helpers import write_lines, get_temp_filepath, generalized_lru_cache, read_lines, log_action, mute
from muss.resources.prepare import prepare_laser

ENCODER_PATH = LASER_DIR / 'models/bilstm.93langs.2018-12-26.pt'
BPE_CODES_PATH = LASER_DIR / 'models/93langs.fcodes'


@generalized_lru_cache(maxsize=1)
@clear_cuda_cache
def get_laser_embeddings(
    sentences,
    bpe_codes_path=BPE_CODES_PATH,
    encoder_path=ENCODER_PATH,
    language='en',
    max_tokens=12000,
    normalize_l2=False,
    n_encoding_jobs=10,
):
    prepare_laser()
    from embed import SentenceEncoder  # noqa: E402
    from text_processing import Token, BPEfastApply  # noqa: E402

    def get_laser_encoder(encoder_path, max_tokens=12000):
        return SentenceEncoder(encoder_path, max_sentences=None, max_tokens=max_tokens, cpu=False)

    def encode_file(input_filepath, output_filepath, language, bpe_codes_path):
        tokenized_filepath = get_temp_filepath()
        Token(str(input_filepath), str(tokenized_filepath), lang=language, romanize=True if language == 'el' else False)
        BPEfastApply(str(tokenized_filepath), str(output_filepath), str(bpe_codes_path))
        tokenized_filepath.unlink()

    input_filepath = get_temp_filepath()
    write_lines(sentences, input_filepath)
    with mute():
        with log_action('Tokenizing and applying BPE'):
            parallel_file_encoder = get_parallel_file_preprocessor(
                lambda input_filepath, output_filepath: encode_file(
                    input_filepath, output_filepath, language, bpe_codes_path
                ),
                n_jobs=n_encoding_jobs,
            )
            bpe_filepath = get_temp_filepath()
            parallel_file_encoder(input_filepath, bpe_filepath)
        with log_action('Geting LASER embedding'):
            encoder = get_laser_encoder(encoder_path, max_tokens=max_tokens)
            embeddings = encoder.encode_sentences(read_lines(bpe_filepath))
            input_filepath.unlink()
            bpe_filepath.unlink()
            assert embeddings.shape[0] == len(sentences)
    del encoder
    if normalize_l2:
        embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=1), axis=1)
    return embeddings
