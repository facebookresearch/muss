# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy

from muss.utils.helpers import print_running_time
from muss.utils.submitit import get_executor, print_job_id
from muss.utils.training import print_function_name, print_args, print_result
from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.fairseq.base import get_fairseq_exp_dir
from muss.mining.training import (
    get_transformer_kwargs,
    get_bart_kwargs,
    get_mbart_kwargs,
    get_score_rows,
    get_all_baseline_rows,
)
from muss.resources.datasets import mix_datasets, create_smaller_dataset


def get_mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    a = 1.0 * data
    a = a[~np.isnan(a)]
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def get_formatted_mean_and_confidence_interval(array, confidence=0.95):
    array = np.array(array)
    mean = np.mean(array)
    confidence_interval = get_mean_confidence_interval(array, confidence=confidence)
    return f"{mean:.2f}±{confidence_interval:.2f} ({array.size})"


print('WARNING: This script probably won\'t work flawlessly out of the box, but it should be easy to understand how to make it work for your use case.')
wikilarge = 'wikilarge_detokenized-wo_turkcorpus'
uts_en_1bq_paraphrases = 'uts_en_query-9c9aa1cf05b77f6cd018a159bd9eaeb0_db-9c9aa1cf05b77f6cd018a159bd9eaeb0_topk-8_nprobe-16_density-0.6_distance-0.05_levenshtein-0.2_simplicity-0.0-wo_turkcorpus'  # noqa: E501
uts_fr_1bq_paraphrases = 'uts_fr_query-0b11603ee8cf563f443458c204130bb1_db-0b11603ee8cf563f443458c204130bb1_topk-8_nprobe-16_density-0.6_distance-0.05_levenshtein-0.2_simplicity-0.0-wo_alector'  # noqa: E501
uts_es_1bq_paraphrases = 'uts_es_query-bb318e13fdbc98cf38b9ef4430aae1a1_db-bb318e13fdbc98cf38b9ef4430aae1a1_topk-8_nprobe-16_density-0.6_distance-0.05_levenshtein-0.2_simplicity-0.0-wo_simplext_data_for_journal_saggion_newsela'  # noqa: E501

kwargs_dict = {
    # English table
    'transformer_wikilarge_wo_turkcorpus': get_transformer_kwargs(dataset=wikilarge, language='en', use_access=False),
    'bart_access_wikilarge_wo_turkcorpus': get_bart_kwargs(dataset=wikilarge, language='en', use_access=True),
    'bart_access_mix_wikilarge_wo_turkcorpus_uts_en_1bq_paraphrases_wo_turkcorpus': get_bart_kwargs(
        dataset=mix_datasets([wikilarge, uts_en_1bq_paraphrases]), language='en', use_access=True
    ),
    'transformer_uts_en_1bq_paraphrases_wo_turkcorpus': get_transformer_kwargs(
        dataset=uts_en_1bq_paraphrases, language='en', use_access=False
    ),
    'mbart_access_uts_en_1bq_paraphrases_wo_turkcorpus': get_mbart_kwargs(
        dataset=uts_en_1bq_paraphrases, language='en', use_access=True
    ),
    'bart_access_uts_en_1bq_paraphrases_wo_turkcorpus': get_bart_kwargs(
        dataset=uts_en_1bq_paraphrases, language='en', use_access=True
    ),
    # French table
    'transformer_uts_fr_1bq_paraphrases_wo_alector': get_transformer_kwargs(
        dataset=uts_fr_1bq_paraphrases, language='fr', use_access=False
    ),
    'mbart_access_uts_fr_1bq_paraphrases_wo_alector': get_mbart_kwargs(
        dataset=uts_fr_1bq_paraphrases, language='fr', use_access=True
    ),
    # Spanish table
    'transformer_uts_es_1bq_paraphrases_wo_simplext': get_transformer_kwargs(
        dataset=uts_es_1bq_paraphrases, language='es', use_access=False
    ),
    'mbart_access_uts_es_1bq_paraphrases_wo_simplext': get_mbart_kwargs(
        dataset=uts_es_1bq_paraphrases, language='es', use_access=True
    ),
    # Ablation size of data
    'bart_access_uts_en_1bq_paraphrases_1k': get_bart_kwargs(
        dataset=create_smaller_dataset(uts_en_1bq_paraphrases, 1000), language='en', use_access=True
    ),
    'bart_access_uts_en_1bq_paraphrases_10k': get_bart_kwargs(
        dataset=create_smaller_dataset(uts_en_1bq_paraphrases, 10000), language='en', use_access=True
    ),
    'bart_access_uts_en_1bq_paraphrases_100k': get_bart_kwargs(
        dataset=create_smaller_dataset(uts_en_1bq_paraphrases, 100000), language='en', use_access=True
    ),
    # Ablation BART + ACCESS
    'bart_uts_en_1bq_paraphrases_wo_turkcorpus': get_bart_kwargs(
        dataset=uts_en_1bq_paraphrases, language='en', use_access=False
    ),
    'transformer_access_uts_en_1bq_paraphrases_wo_turkcorpus': get_transformer_kwargs(
        dataset=uts_en_1bq_paraphrases, language='en', use_access=True
    ),
}

jobs_dict = defaultdict(list)
for exp_name, kwargs in tqdm(kwargs_dict.items()):
    executor = get_executor(
        cluster='local',
        slurm_partition='priority',
        submit_decorators=[print_function_name, print_args, print_job_id, print_result, print_running_time],
        timeout_min=2 * 24 * 60,
        gpus_per_node=kwargs['train_kwargs']['ngpus'],
        nodes=1,
        slurm_constraint='volta32gb',
        name=exp_name,
    )
    for i in range(5):
        job = executor.submit(fairseq_train_and_evaluate_with_parametrization, **kwargs)
        jobs_dict[exp_name].append(job)
[job.result() for jobs in jobs_dict.values() for job in jobs]

# Evaluation
table = []
for exp_name, jobs in tqdm(jobs_dict.items()):
    for job in jobs:
        exp_dir = get_fairseq_exp_dir(job.job_id)
        kwargs = job.submission().kwargs
        table.extend(get_score_rows(exp_dir, kwargs, additional_fields={'exp_name': exp_name, 'job_id': job.job_id}))
table.extend(print_running_time(get_all_baseline_rows)())
df_scores = pd.DataFrame(table)


def mean(arr):
    if len(arr) not in [8, 10]:  # Hack for Reference rows
        arr = arr[-5:]
    return get_formatted_mean_and_confidence_interval(arr)


pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', 100)
print(
    df_scores.groupby(['language', 'dataset', 'phase', 'exp_name'])
    .agg([mean])[['sari', 'bleu', 'fkgl']]
    .sort_values(by=['language', 'dataset', 'phase', ('sari', 'mean')])
)
