## ---------------------------- import libraries ---------------------------- ##
import numpy as np
import pandas as pd
import collections
import random
import math
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.utils import resample
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score

# import matplotlib.pyplot as plt

## --------------------------- ----------------------------- ##


def binarize_multi_labels(data, feat_name):
    """
    binarinize feature consisting of multiple labels
    @feat_name[str]: name of the feature to be processed
    """
    
    # split each field into list of labels
    labels = data[feat_name].str.split(',')
    
    # construct dummy variables 
    mlb = MultiLabelBinarizer()
    labels_dm = mlb.fit_transform(labels)
    labels_df = pd.DataFrame(labels_dm, index=data.index).add_prefix('usertag_')
    
    # concat dummy variables to data and drop original feature
    data = pd.concat([data, labels_df], axis=1)
    data = data.drop(feat_name, axis=1)
    
    return data
    

def add_features(data):
    """
    add hand-crafted features: slotarea, system , browser, usertag_x, slotprice_x
    """
    
    # compute area of ad slot
    data['slotarea'] = data['slotheight'] * data['slotwidth']
    
    # split useragent into system and browser
    data[['system', 'browser']] = data['useragent'].str.split('_', expand=True)
    data = data.drop('useragent', axis=1)
    
    # binarinize usertag[list] to usertag_0, ...usertag_n
    data = binarize_multi_labels(data, 'usertag')
    
    return data



def cont_to_binarized_bin(data, cont_to_bin_cols, bins):
    """
    binarinize indicies of bins to which each continuous value in the input belongs
    @cont_col[list]: list of name for the continuous feature to be converted
    @bins[list]: list of array of bin boundaries, defined to make each bin has similar amount
    """
    
    for i, cont_col in enumerate(cont_to_bin_cols):
        data[cont_col + '_bin'] = np.digitize(data[cont_col], bins=bins[i])
        data = pd.get_dummies(data, columns= [cont_col+'_bin'])
    
    return data



def min_max_scaling(data, scale_cols):
    """
    process continuous/ordinal features using min-max scaling,
    not suitable for feature with outlier
    """
    
    min_max_scaler = MinMaxScaler()
    data[scale_cols] = min_max_scaler.fit_transform(data[scale_cols])
    
    return data



def split_data(merged, train, valid, test):
    """
    split dataset into train, validation and test
    """
    
    train_data = merged[:train.shape[0]]
    valid_data = merged[train.shape[0]: train.shape[0]+valid.shape[0]]
    test_data  = merged[-test.shape[0]:]
    
    return train_data, valid_data, test_data
    

    
def downsampling_majority_class(data, class_ratio = 0.05, seed=500):

    # Display old class counts
    print('The initial dataset has following sizes for each class:')
    print(data.click.value_counts())

    # Separate majority and minority classes
    data_majority = data[data.click == 0]
    data_minority = data[data.click == 1]

    print('Minority class is %.2f%% of initial sample size.' % (len(data_minority)/len(data)*100))

    # Samples to be drawn
    len_majority = math.floor(len(data)*(len(data_minority)/len(data))/class_ratio-len(data_minority))

    # Downsample
    data_majority_downsampled = resample(data_majority,
                                         replace=False,
                                         n_samples=len_majority,
                                         random_state=seed)

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([data_minority, data_majority_downsampled])

    # Display new class counts
    print('New dataset has following sizes for each class:')
    print(df_downsampled.click.value_counts())
    print('Minority class is %.2f%% of total sample size.' % (len(data_minority)/len(df_downsampled)*100))

    return df_downsampled


    
def preprocessing(irrelevant_cols, cont_cols, cat_cols, cont_to_bin_cols, bins, train, valid, test):
    """
    Pipeline for data preprocessing
    """
    
    # concat train, valid, test datasets for data preprocessing
    data = pd.concat([train, valid, test], axis=0)

    # drop irrelevant columns
    data = data.drop(irrelevant_cols, axis=1)

    # add hand-crafted features: slotarea, system , browser, usertag_x
    data = add_features(data)
    
    # binarinize indicies of bins to which each continuous value in the input belongs
    data = cont_to_binarized_bin(data, cont_to_bin_cols, bins)

    # process continuous/ordinal features
    ## min-max scaling
    data = min_max_scaling(data, cont_cols)

    # process non-ordinal features: one hot encoding
    # 'slotid' is not included
    data = pd.get_dummies(data, columns=cat_cols)

    # split concated dataset into train, validation and test
    train_data, valid_data, test_data = split_data(data, train, valid, test)
    
    return data, train_data, valid_data, test_data



def preprocessing_t(irrelevant_cols, cont_cols, cat_cols, cont_to_bin_cols, bins, train, valid, test):
    """
    Pipeline for data preprocessing for tree-based methods
    """
    
    # concat train, valid, test datasets for data preprocessing
    data = pd.concat([train, valid, test], axis=0)

    # drop irrelevant columns
    data = data.drop(irrelevant_cols, axis=1)

    # add hand-crafted features: slotarea, system , browser, usertag_x
    data = add_features(data)
    
    # binarinize indicies of bins to which each continuous value in the input belongs
    data = cont_to_binarized_bin(data, cont_to_bin_cols, bins)

    # process continuous/ordinal features
    ## min-max scaling
#    data = min_max_scaling(data, cont_cols)

    # process non-ordinal features: one hot encoding
    # 'slotid' is not included
#    data = pd.get_dummies(data, columns=cat_cols)

    # split concated dataset into train, validation and test
#    train_data, valid_data, test_data = split_data(data, train, valid, test)
    
#    return data, train_data, valid_data, test_data

    return data
    

    
    
  

