# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
from pathlib import Path
import shutil

from muss.utils.resources import download_and_extract

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_DIR / 'experiments'
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
VARIOUS_DIR = RESOURCES_DIR / 'various'
MODELS_DIR = RESOURCES_DIR / 'models'
TOOLS_DIR = RESOURCES_DIR / 'tools'
SUBMITIT_LOGS_DIR = EXP_DIR / 'submitit_logs/'
SUBMITIT_JOB_DIR_FORMAT = SUBMITIT_LOGS_DIR / '%j'
TENSORBOARD_LOGS_DIR = EXP_DIR / 'tensorboard_logs'
# TODO: Move this to setup or add the folders to the git repo
for dir_path in [DATASETS_DIR, VARIOUS_DIR, MODELS_DIR, TOOLS_DIR, TENSORBOARD_LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
FASTTEXT_EMBEDDINGS_DIR = Path(VARIOUS_DIR) / 'fasttext-vectors/'
LASER_DIR = TOOLS_DIR / 'LASER'

LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']


def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset


def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    filename = f'{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename


def get_filepaths_dict(dataset):
    return {
        (phase, language): get_data_filepath(dataset, phase, language) for phase, language in product(PHASES, LANGUAGES)
    }


def get_fasttext_embeddings_path(language='en'):
    fasttext_embeddings_path = FASTTEXT_EMBEDDINGS_DIR / f'cc.{language}.300.vec'
    if not fasttext_embeddings_path.exists():
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz'
        fasttext_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(download_and_extract(url)[0], fasttext_embeddings_path)
    return fasttext_embeddings_path
