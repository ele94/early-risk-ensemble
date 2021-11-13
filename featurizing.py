import pickle
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import *
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np

oversample = True
calculate_feats = True
normalize = True
discretize = True
exclude_feats = []
discretize_size = 10


def featurize(calculate_feats=False, normalize=False, discretize=False, scale=False, 
              discretize_size=10, dis_strategy="kmeans", include_feats=[], train_users=None, test_users=None, save=False):
    
    nssi_corpus = load_nssi_corpus()
    
    logger("Featurizing calculate_feats={}, normalize={}, discretize={}, discretize_size={}, include_feats={}".format(calculate_feats, normalize, discretize, discretize_size, include_feats))
    
    import numpy
    import tensorflow
    import sys

    from numpy.random import seed
    seed(42)
    tensorflow.random.set_seed(42) 
    logger("Initialized numpy random and tensorflow random seed at 42")
    
    if calculate_feats:
        if train_users is None or test_users is None:
            train_users = load_pickle(pickle_path, "train_users.pkl")
            test_users = load_pickle(pickle_path, "test_users.pkl")
        X_train = train_users["clean_text"]
        X_test = test_users["clean_text"]
        
        logger("Data size: {}, {}".format(X_train.shape[0], train_users.shape[0]))
        logger("Data size: {}, {}".format(X_test.shape[0], test_users.shape[0]))
        
        feats_train = calculate_features(X_train, train_users, nssi_corpus, include_feats)
        feats_test = calculate_features(X_test, test_users, nssi_corpus, include_feats)
        
        save_pickle(pickle_path, "feats_train_original.pkl", feats_train)
        save_pickle(pickle_path, "feats_test_original.pkl", feats_test)
        
    else:
        feats_train = load_pickle(pickle_path, "feats_train_original.pkl")
        feats_test = load_pickle(pickle_path, "feats_test_original.pkl")
        
    #logger(feats_train.describe())
    #logger(feats_test.describe())
        
    #feats_train, feats_test = select_features(feats_train, feats_test, exclude_feats=exclude_feats, 
    #                                          normalize=normalize, discretize=discretize,discretize_size=discretize_size)
    
    if normalize:
        logger("Normalizing features")
        feats_train = normalize_features(feats_train)
        feats_test = normalize_features(feats_test)
    if scale: 
        logger("Scaling features")
        feats_train, feats_test = scale_features(feats_train, feats_test)
    if discretize:
        logger("Discretizing")
        feats_train, feats_test = discretize_features(feats_train, feats_test, size=discretize_size, strategy=dis_strategy)
    
    if save:
        logger("Saving variables to memory")
        save_pickle(pickle_path, "feats_train.pkl", feats_train)
        save_pickle(pickle_path, "feats_test.pkl", feats_test)
    
    return feats_train, feats_test
        


def scale_features(feats_train, feats_test):
    
    #train_features = np.array(feats_train)
    #test_features = np.array(feats_test)
    scaler = StandardScaler()
    train_features = scaler.fit_transform(feats_train)
    
    test_features = scaler.transform(feats_test)

    train_features = np.clip(train_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)
    
    logger('Training features shape: {}'.format(train_features.shape))
    logger('Test features shape: {}'.format(test_features.shape))
    
    return train_features, test_features




def calculate_features(X, users, nssi_corpus, include_features=[]):
    
    feats = pd.DataFrame()
    #text len
    feats['char_count'] = X.map(len)
    #word count
    feats['word_count'] = X.map(lambda x: len(x.split()))
    
    #special features
    #first prons
    if 'first_prons' in include_features:
        logger("Calculating first prons")
        reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
        feats['first_prons'] = X.map(lambda x: len(re.findall(reg, x)))
    # sentiment analysis
    if 'sentiment' in include_features:
        logger("Calculating sentiment")
        sid = SentimentIntensityAnalyzer()
        feats['sentiment'] = X.map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    
    if 'nssi' in include_features:
        logger("Calculating NSSI words")
        # nssi words
        for key, values in nssi_corpus.items():
            feats[key] = users['stems'].map(lambda x: sum((' '.join(x)).count(word) for word in values))
        
    return feats


def load_nssi_corpus():

    with open("/datos/erisk/ml/data/nssicorpus.txt", 'r') as file:
        nssi_corpus_original = file.read()

    nssi_corpus = nssi_corpus_original.replace('*', '')
    nssi_corpus = nssi_corpus.replace("Methods of NSSI", '')
    nssi_corpus = nssi_corpus.replace("NSSI Terms", '')
    nssi_corpus = nssi_corpus.replace("Instruments Used", '')
    nssi_corpus = nssi_corpus.replace("Reasons for NSSI", '')

    keys = ["methods", "terms", "instruments", "reasons"]

    nssi_corpus = nssi_corpus.split(':')
    nssi_corpus.remove('')
    nssi_corpus = [corpus.split("\n") for corpus in nssi_corpus]
    new_nssi_corpus = {}
    for idx, corpus in enumerate(nssi_corpus):
        new_list = [word for word in corpus if word != ""]
        new_nssi_corpus[keys[idx]] = new_list

    return new_nssi_corpus

def select_features(feats_train, feats_test, exclude_feats=[], normalize=False, discretize=False, discretize_size=10):
    feats_train_ret = feats_train.copy()
    feats_test_ret = feats_test.copy()
    
    for feat in exclude_feats:
        feats_train_ret.drop(feat, inplace=True, axis=1)
        feats_test_ret.drop(feat, inplace=True, axis=1)
    
    if normalize:
        feats_train_ret = normalize_features(feats_train_ret)
        feats_test_ret = normalize_features(feats_test_ret)
        
    if discretize:
        feats_train_ret, feats_test_ret = discretize_features(feats_train_ret, feats_test_ret, size=discretize_size)
    else:
        feats_train_ret = feats_train_ret.values
        feats_test_ret = feats_test_ret.values
    
    return feats_train_ret, feats_test_ret

normalize_exceptions = ['char_count', 'word_density']

def normalize_features(feats):
    text_length = feats["char_count"]
    
    norm_feats = feats.copy()
    for feature in feats.columns:
        if feature not in normalize_exceptions:
            norm_feats[feature] = feats[feature] / text_length
            
    return norm_feats

from sklearn.preprocessing import KBinsDiscretizer

def discretize_features(train_feats, test_feats, size=10, strategy='kmeans', encode='onehot-dense'):
    est = KBinsDiscretizer(n_bins=size, encode=encode, strategy=strategy)
    train = est.fit_transform(train_feats)
    test = est.transform(test_feats)

    return train, test