import re
import string

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PUNCTUATION_TO_REMOVE = list(string.punctuation)
PUNCTUATION_TO_LABEL = [',', '.', ';', ':']
for symbol_ in ['#', '@'] + PUNCTUATION_TO_LABEL:
    PUNCTUATION_TO_REMOVE.remove(symbol_)


def hashtags(x):
    tokens = x.apply(lambda y: y.split())
    hashtag_tokens_list = tokens.apply(lambda y: [token for token in y if token[0]=='#'])
    num_hashtags = hashtag_tokens_list.apply(lambda y: len(y))
    return num_hashtags.to_frame()


def mean_word_length(x):
    tokens = x.apply(lambda y: y.split())
    mean_token_length = np.mean(np.array([len(token) for token in tokens]).reshape(-1,1), axis=1)
    return pd.DataFrame(mean_token_length)


def mentions(x):
    tokens = x.apply(lambda y: y.split())
    mentions_tokens_list = tokens.apply(lambda y: [token for token in y if token[0]=='@'])
    num_mentions = mentions_tokens_list.apply(lambda y: len(y))
    return num_mentions.to_frame()


def tweet_length(x):
    num_characters = x.apply(lambda y: len(y))
    return num_characters.to_frame()


def links(x):
    tokens = x.apply(lambda y: y.split())
    links_token_list = tokens.apply(lambda y: [token for token in y if ('http' in token) or ('www' in token)])
    num_links = links_token_list.apply(lambda y: len(y))
    return num_links.to_frame()


def replace_numbers_and_punctuation(word_list_df):
    series_of_word_lists = word_list_df[0]
    series_of_word_lists = series_of_word_lists.apply(lambda string_list: [word if 'http' not in word else 'URL' for word in string_list])
    series_of_strings = series_of_word_lists.apply(lambda word_list: ' '.join(word_list))
    re_number = re.compile('\d+')
    strings_formatted = series_of_strings.apply(lambda string_: re_number.sub(' number ', string_))
    strings_formatted = strings_formatted.apply(lambda string_: string_.lower())
    for symbol_ in PUNCTUATION_TO_REMOVE:
        strings_formatted = strings_formatted.apply(lambda string_: string_.replace(symbol_, ''))
    for symbol_ in PUNCTUATION_TO_LABEL:
        strings_formatted = strings_formatted.apply(lambda string_: string_.replace(symbol_, ' punctuation '))
    strings_formatted = strings_formatted.apply(lambda string_: string_.replace('#', ' hashtag '))
    strings_formatted = strings_formatted.apply(lambda string_: string_.replace('@', ' mention '))
    return strings_formatted


def get_words(series_of_strings):
    words = series_of_strings.apply(lambda string_: string_.split())
    return words.to_frame()


def remove_stop_words(words_list_ndarray):
    words_list_series = pd.Series(words_list_ndarray.reshape(-1))
    non_stop_words = words_list_series.apply(lambda words_list: [word for word in words_list if word not in ENGLISH_STOP_WORDS])
    return non_stop_words.to_frame()