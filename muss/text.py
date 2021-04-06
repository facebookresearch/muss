# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import re
import unicodedata


SPECIAL_TOKEN_REGEX = r'<[a-zA-Z\-_\d\.]+>'


def extract_special_tokens(sentence):
    '''Remove any number of token at the beginning of the sentence'''
    match = re.match(fr'(^(?:{SPECIAL_TOKEN_REGEX} *)+) *(.*)$', sentence)
    if match is None:
        return '', sentence
    special_tokens, sentence = match.groups()
    return special_tokens.strip(), sentence


@lru_cache(maxsize=100)  # To speed up subsequent calls
def word_tokenize(sentence, language='en'):
    special_tokens, sentence = extract_special_tokens(sentence)
    tokenized_sentence = ' '.join([tok.text for tok in spacy_process(sentence, language=language)])
    if special_tokens != '':
        tokenized_sentence = f'{special_tokens} {tokenized_sentence}'
    return tokenized_sentence


@lru_cache(maxsize=1)
def get_treebank_word_detokenizer():
    # Inline lazy import because importing nltk is slow
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    return TreebankWordDetokenizer()


def nltk_word_detokenize(sentence):
    return get_treebank_word_detokenizer().detokenize(to_words(sentence))


@lru_cache()
def get_moses_detokenizer(language):
    # Inline lazy import because importing sacremoses is slow
    from sacremoses import MosesDetokenizer

    return MosesDetokenizer(lang=language)


def moses_word_detokenize(sentence, language='en'):
    return get_moses_detokenizer(language).detokenize(sentence.split())


def word_detokenize(sentence, backend='moses', **kwargs):
    detokenize_function = {'moses': moses_word_detokenize, 'nltk': nltk_word_detokenize}[backend]
    return detokenize_function(sentence, **kwargs)


def to_words(sentence):
    return sentence.split()


@lru_cache(maxsize=1)
def get_sentence_tokenizer(language='en'):
    # Inline lazy import because importing nltk is slow
    import nltk

    language = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish',
        'it': 'italian',
        'de': 'german',
    }[language]
    return nltk.data.load(f'tokenizers/punkt/{language}.pickle')


def to_sentences(text, language='en'):
    text = ' '.join(text.split('\n'))  # Remove newlines
    return get_sentence_tokenizer(language).tokenize(text)


def remove_multiple_whitespaces(text):
    return re.sub(r'  +', ' ', text)


# Adapted from the following scripts:
# https://github.com/XingxingZhang/dress/blob/master/dress/scripts/readability/syllables_en.py
# https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/readability/syllables_en.py
"""
Fallback syllable counter
This is based on the algorithm in Greg Fast's perl module
Lingua::EN::Syllable.
"""

specialSyllables_en = """tottered 2
chummed 1
peeped 1
moustaches 2
shamefully 3
messieurs 2
satiated 4
sailmaker 4
sheered 1
disinterred 3
propitiatory 6
bepatched 2
particularized 5
caressed 2
trespassed 2
sepulchre 3
flapped 1
hemispheres 3
pencilled 2
motioned 2
poleman 2
slandered 2
sombre 2
etc 4
sidespring 2
mimes 1
effaces 2
mr 2
mrs 2
ms 1
dr 2
st 1
sr 2
jr 2
truckle 2
foamed 1
fringed 2
clattered 2
capered 2
mangroves 2
suavely 2
reclined 2
brutes 1
effaced 2
quivered 2
h'm 1
veriest 3
sententiously 4
deafened 2
manoeuvred 3
unstained 2
gaped 1
stammered 2
shivered 2
discoloured 3
gravesend 2
60 2
lb 1
unexpressed 3
greyish 2
unostentatious 5
"""

fallback_cache = {}

fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou", "sia$", ".ely$"]

fallback_addsyl = [
    "ia",
    "riet",
    "dien",
    "iu",
    "io",
    "ii",
    "[aeiouy]bl$",
    "mbl$",
    "[aeiou]{3}",
    "^mc",
    "ism$",
    "(.)(?!\\1)([aeiouy])\\2l$",
    "[^l]llien",
    "^coad.",
    "^coag.",
    "^coal.",
    "^coax.",
    "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]",
    "dnt$",
]

# Compile our regular expressions
for i in range(len(fallback_subsyl)):
    fallback_subsyl[i] = re.compile(fallback_subsyl[i])
for i in range(len(fallback_addsyl)):
    fallback_addsyl[i] = re.compile(fallback_addsyl[i])


def _normalize_word(word):
    return word.strip().lower()


# Read our syllable override file and stash that info in the cache
for line in specialSyllables_en.splitlines():
    line = line.strip()
    if line:
        toks = line.split()
        assert len(toks) == 2
        fallback_cache[_normalize_word(toks[0])] = int(toks[1])


@lru_cache(maxsize=10)
def get_spacy_model(language='en', size='md'):
    # Inline lazy import because importing spacy is slow
    import spacy

    if language == 'it' and size == 'md':
        print('Model it_core_news_md is not available for italian, falling back to it_core_news_sm')
        size = 'sm'
    model_name = {
        'en': f'en_core_web_{size}',
        'fr': f'fr_core_news_{size}',
        'es': f'es_core_news_{size}',
        'it': f'it_core_news_{size}',
        'de': f'de_core_news_{size}',
    }[language]
    return spacy.load(model_name)  # python -m spacy download en_core_web_sm


@lru_cache(maxsize=10 ** 6)
def spacy_process(text, language='en', size='md'):
    return get_spacy_model(language=language, size=size)(str(text))


def _get_named_entities_spacy(text, language):
    # List of tuples of the form: [('LABEL', (start_char, end_char)), ...] where (start_char, end_char) are the start
    # and end character indexes of the words forming a given named entity
    return [(ent.label_, (ent.start_char, ent.end_char)) for ent in spacy_process(text, language=language).ents]


def _get_named_entities_nltk(text, language):
    assert False, 'NLTK named entity retrieval does not work anymore (need to find the indexes of NEs in the text)'
    assert language == 'en'
    # Inline lazy import because importing nltk is slow
    import nltk

    return [
        (chunk.label(), ' '.join(c[0] for c in chunk))
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
        if hasattr(chunk, 'label')
    ]


def get_named_entities(text, language='en', backend='spacy'):
    if backend == 'spacy':
        return _get_named_entities_spacy(text, language=language)
    elif backend == 'nltk':
        return _get_named_entities_nltk(text, language=language)
    else:
        raise NotImplementedError(f'Invalid backend "{backend}"')


def yield_sentence_concatenations(text, min_length=10, max_length=300, language='en'):
    sentences = to_sentences(text, language=language)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences) + 1):
            concatenation = ' '.join(sentences[i:j])
            if len(concatenation) < min_length:
                continue
            if len(concatenation) > max_length:
                break
            yield concatenation


def normalize_unicode(text):
    # Normalize characters such as accented characters
    return unicodedata.normalize('NFKC', text)


@lru_cache(maxsize=1)
def get_spacy_tokenizer(language='en'):
    return get_spacy_model(language=language).Defaults.create_tokenizer(get_spacy_model(language=language))


def get_spacy_content_tokens(text, language='en'):
    def is_content_token(token):
        return not token.is_stop and not token.is_punct and token.ent_type_ == ''  # Not named entity

    return [token for token in get_spacy_tokenizer(language=language)(text) if is_content_token(token)]


def get_content_words(text, language='en'):
    return [token.text for token in get_spacy_content_tokens(text, language=language)]


def truncate(sentence, truncate_prop=0.2, language='en'):
    words = to_words(word_tokenize(sentence))
    if words[-1] == '.':
        words = words[:-1]
    n_truncated_words = round(truncate_prop * len(words))
    words = words[:-n_truncated_words] + ['.']
    return word_detokenize(' '.join(words))
