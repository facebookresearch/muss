# Multilingual Unsupervised Sentence Simplification

Code to reproduce experiments in "MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases"

## Prerequisites

Linux with python 3.6 or above.

## Installing

```
pip install .  # Install package
pip install -r requirements.txt  # Install dependencies
```

## How to use
Some scripts might still contain a few bugs, if you notice anything wrong, feel free to open an issue or submit a Pull Request.

### Simplify sentences from a file using pretrained models
```python
# English
python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined
# French
python scripts/simplify.py scripts/examples.fr --model-name muss_fr_mined
# French
python scripts/simplify.py scripts/examples.es --model-name muss_es_mined
```

### Mine the data
```python
python scripts/mine_sequences.py
```

### Train the models
```python
python scripts/train_model.py
```

### Evaluate simplifications
Please head over to [EASSE](https://github.com/feralvam/easse/) for Sentence Simplification evaluation.


## License

The MUSS license is CC-BY-NC. See the [LICENSE](LICENSE) file for more details.

## Authors

* **Louis Martin** ([louismartincs@gmail.com](mailto:louismartincs@gmail.com))


## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}
```
