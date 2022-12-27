import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

def read_data():
    '''
    Read in the training data.

    Returns
    -------
    data : Pandas dataframe
        Returns a dataframe containing the training data.
    '''
    data = pd.read_csv('../../notebooks/drugsComTrain_raw.tsv',sep='\t')
    return data


def filter_review(review, remove_stop_words=True):
    '''
    Remove unnecessary words and characters from the drug reviews.

    Parameters
    ----------
    review : str
        Drug review string.

    remove_stop_words : bool
        Whether to remove the stopwords.
    '''
    characters = '!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r'

    review = review.lower()
    review = review.strip('"')
    review = review.replace("'","")
    review = review.replace('&#039;', "'")
    review = ''.join(list(filter(lambda char: char not in characters, review)))
    review = review.split()

    if remove_stop_words:
        stop_word_list = stopwords.words('english')
        stop_word_list.remove('no')
        review = list(filter(lambda word: word not in stop_word_list, review))

    return review


def create_master_list(reviews, remove_stop_words=True):
    '''
    Create a list of all of the reviews with stop words and unnecessary
    characters removed.

    Parameters
    ----------
    reviews : list
        List of reviews.

    remove_stop_words : bool
        Whether to remove the stop words in each review.
    '''
    master_list = [filter_review(review, remove_stop_words=remove_stop_words) for review in reviews]

    return master_list


def create_tokenizer(reviews, remove_stop_words=True):
    '''
    Create a tokenizer trained to convert reviews to tokens.

    Parameters
    ----------
    reviews : list
        List of reviews.

    remove_stop_words : bool
        Whether to remove the stop words in each review.
    '''
    tokenizer = Tokenizer(oov_token='unk')
    master_list = create_master_list(reviews, remove_stop_words=remove_stop_words)
    tokenizer.fit_on_texts(master_list)

    return tokenizer


def tokenize_reviews(reviews, remove_stop_words=True):
    '''
    Create a tokenizer trained to convert reviews to tokens, and tokenize the
    reviews. Returns tokenized reviews and the tokenizer used.

    Parameters
    ----------
    reviews : list
        List of reviews.

    remove_stop_words : bool
        Whether to remove the stop words in each review.
    '''

    tokenizer = create_tokenizer(reviews, remove_stop_words=remove_stop_words)

    reviews_tokenized = tokenizer.texts_to_sequences(reviews)

    return reviews_tokenized, tokenizer




def testing():
    print(read_data())
    print(create_tokenizer(['hi i am a food person']))
    print(tokenize_reviews(['hi i am a food person']))

if __name__ == '__main__':
    print(testing())
