import unicodedata, re, os, io
import tensorflow as tf
import numpy as np

def preprocess_sentence(w, add_token=True):
    w = re.sub(r"([?.!,'¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?'.!,¿ıüçğşö]+", " ", w)
    w = w.rstrip().strip()
    if add_token: w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples, add_token=True):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w, add_token) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def add_unknown_to_vocab(lang):
  index_word, word_index = {1:'<unkown>'}, {'<unkown>': 1}
  for i in lang.index_word:
    index_word[i+1] = lang.index_word[i]
    word_index[lang.index_word[i]] = i+1
  return index_word, word_index


def remove_words(tokenizer, num_words):
    tokenizer.index_word = {k: tokenizer.index_word[k] for k in list(tokenizer.index_word)[:num_words]}
    tokenizer.word_index = {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:num_words]}
    return tokenizer


def tokenize(lang, maxlen, num_words):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  lang_tokenizer.index_word, lang_tokenizer.word_index = add_unknown_to_vocab(lang_tokenizer)
  lang_tokenizer = remove_words(lang_tokenizer, num_words)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=maxlen)
  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None, maxlen=50, num_words=None):
    inp_lang, targ_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang, maxlen, num_words)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang, maxlen, num_words)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
