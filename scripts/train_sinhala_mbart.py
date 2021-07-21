
from muss.fairseq.main import fairseq_train_and_evaluate_with_parametrization
from muss.mining.training import get_mbart_kwargs


sin15M = 'sin15M'
kwargs = get_mbart_kwargs(dataset=sin15M, language='si', use_access=True)
kwargs['train_kwargs']['ngpus'] = 1  # Set this from 8 to 1 for local training
kwargs['train_kwargs']['max_tokens'] = 512  # Lower this number to prevent OOM
result = fairseq_train_and_evaluate_with_parametrization(**kwargs)