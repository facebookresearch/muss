
from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_mbart_kwargs


newsela = 'newsela'
kwargs = get_mbart_kwargs(dataset=newsela, language='si', use_access=False)
kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 32  # Lower this number to prevent OOM
result = fairseq_train_and_evaluate_with_parametrization(**kwargs)