# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from glob import glob
import os
import shutil
import sys

from muss.utils.helpers import run_command, create_directory_or_skip
from muss.preprocessing import replace_lrb_rrb_file, normalize_punctuation
from muss.utils.resources import download_and_extract, add_newline_at_end_of_file, git_clone, download
from muss.resources.paths import get_dataset_dir, get_data_filepath, PHASES, LASER_DIR
from muss.resources.datasets import apply_line_function_to_dataset
from muss.text import normalize_unicode, word_detokenize


def prepare_wikilarge():
    print('WikiLarge')
    dataset = 'wikilarge'  # dataset = wikismall works as well
    with create_directory_or_skip(get_dataset_dir(dataset)):
        url = 'https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2'
        extracted_path = download_and_extract(url)[0]
        # Process
        print('Processing...')
        # Only rename files and put them in local directory architecture
        # FIXME: Wikilarge validations set only has 992 sentences
        for phase in PHASES:
            for (old_language_name, new_language_name) in [('src', 'complex'), ('dst', 'simple')]:
                old_path_glob = os.path.join(extracted_path, dataset, f'*.ori.{phase}.{old_language_name}')
                globs = glob(old_path_glob)
                assert len(globs) == 1
                old_path = globs[0]
                new_path = get_data_filepath(dataset, phase, new_language_name)
                shutil.copyfile(old_path, new_path)
                shutil.move(replace_lrb_rrb_file(new_path), new_path)
                add_newline_at_end_of_file(new_path)
    print('Done.')


def prepare_wikilarge_detokenized():
    def normalize_and_detokenize_line(sentence):
        return word_detokenize(normalize_punctuation(normalize_unicode(sentence)), language='en')

    new_dataset = 'wikilarge_detokenized'
    with create_directory_or_skip(new_dataset):
        prepare_wikilarge()
        apply_line_function_to_dataset(normalize_and_detokenize_line, 'wikilarge', new_dataset)


def prepare_asset():
    print('ASSET')
    dataset = 'asset'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        for phase in ('valid', 'test'):
            for i in range(10):
                for (old_language_name, new_language_name) in [('orig', 'complex'), (f'simp.{i}', f'simple.{i}')]:
                    url = f'https://raw.githubusercontent.com/facebookresearch/asset/master/dataset/asset.{phase}.{old_language_name}'
                    old_path = download(url)
                    new_path = get_data_filepath(dataset, phase, new_language_name)
                    shutil.copyfile(old_path, new_path)
                    add_newline_at_end_of_file(new_path)
    print('Done.')


def prepare_laser():
    os.environ['LASER'] = str(LASER_DIR)
    for path in [LASER_DIR / 'source', LASER_DIR / 'source/lib']:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    if not LASER_DIR.exists():
        git_clone('https://github.com/facebookresearch/LASER.git', LASER_DIR)
        run_command(f'cd {LASER_DIR} && bash ./install_models.sh && bash ./install_external_tools.sh')
