# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().strip().split('\n')

setup(
    name='muss',
    version='1.0',
    description='MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Louis Martin',
    author_email='louismartincs@gmail.com',
    url = 'https://github.com/facebookresearch/muss',
    packages=find_packages('muss'),
    install_requires=requirements,
)
