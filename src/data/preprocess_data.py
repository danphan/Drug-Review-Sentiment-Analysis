import numpy as np
import pandas as pd
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append('src/data')

import preprocess_reviews as pr


def retrieve_training_data():
    '''
    Retrieve the training data.

    Returns
    -------
    data : Pandas dataframe
        Returns a dataframe containing the training data.
    '''
    data = pd.read_csv('notebooks/drugsComTrain_raw.tsv', sep='\t')
    return data


def convert_date_to_int(date):
    '''
    Convert a date to int of the number of days since January 1, 2000.

    Parameters
    ----------
    date : date object formatted as 'Month day, year'
        The date to be converted
    '''
    date = datetime.strptime(date, '%B %d, %Y')
    start_date = datetime.strptime('January 1, 2000', '%B %d, %Y')
    return int((date - start_date).days)


def prepare_dates(data):
    '''
    Parameters
    ----------
    data : pandas Dataframe
        The review data.

    remove_stop_words : bool
        Whether to remove the stopwords.
    '''
    # Create a number of days since Jan 1, 2000 feature
    data['no_of_days'] = data['date'].apply(convert_date_to_int)

    return data


def preprocess_data(train_data, val_data, use_rating=True,
                    remove_stop_words=True):
    '''
    Parameters
    ----------
    train_data : pandas Dataframe
        The training data.

    val_data : pandas Dataframe
        The validation data.

    use_rating : bool
        If True, the rating is used as a feature. If False, the usefulCount is
        used as a feature.

    remove_stop_words : bool
        Whether to remove the stopwords.
    '''
    # Create a number of days since Jan 1, 2000 feature
    train_data = prepare_dates(train_data)
    val_data = prepare_dates(val_data)

    # Determine which features to scale
    if use_rating == True:
        scaled_features = ['no_of_days', 'rating']
    else:
        scaled_features = ['no_of_days', 'usefulCount']

    # Scale the training and validation data (if given)
    scaler = StandardScaler()
    train_data[scaled_features] = scaler.fit_transform(train_data[scaled_features])
    val_data[scaled_features] = scaler.transform(val_data[scaled_features])



    return train_data[scaled_features], val_data[scaled_features]


def testing():
    df = retrieve_training_data()

    train_data, val_data = train_test_split(df, test_size=0.2)
    print(preprocess_data(train_data, val_data, use_rating=True))

    # print(convert_date_to_int('January 2, 2000'))
    # df['date_int'] = df['date'].apply(convert_date_to_int)
    # print(df['date'].head())
    # print(df.dtypes)


if __name__ == '__main__':
    testing()
