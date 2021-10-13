import pickle
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils import *

oversample = True
calculate_feats = True
normalize = True
discretize = True
exclude_feats = []
discretize_size = 10


def featurize(calculate_feats=False, normalize=False, discretize=False, discretize_size=10, exclude_feats=[]):
    
    logger("Featurizing calculate_feats={}, normalize={}, discretize={}, discretize_size={}, exclude_feats={}".format(calculate_feats, normalize, discretize, discretize_size, exclude_feats))
    
    import numpy
    import tensorflow
    import sys

    from numpy.random import seed
    seed(42)
    tensorflow.random.set_seed(42) 
    logger("Initialized numpy random and tensorflow random seed at 42")
    
    if calculate_feats:
        train_users = load_pickle(pickle_path, "train_users.pkl")
        test_users = load_pickle(pickle_path, "test_users.pkl")
        X_train = train_users["clean_text"]
        X_test = test_users["clean_text"]
        
        feats_train = calculate_features(X_train, train_users)
        feats_test = calculate_features(X_test, test_users)
        
        save_pickle(pickle_path, "feats_train_original.pkl", feats_train)
        save_pickle(pickle_path, "feats_test_original.pkl", feats_test)
        
    else:
        feats_train = load_pickle(pickle_path, "feats_train_original.pkl")
        feats_test = load_pickle(pickle_path, "feats_test_original.pkl")
        
    feats_train, feats_test = select_features(feats_train, feats_test, exclude_feats=exclude_feats, 
                                              normalize=normalize, discretize=discretize,discretize_size=discretize_size)
    
    save_pickle(pickle_path, "feats_train.pkl", feats_train)
    save_pickle(pickle_path, "feats_test.pkl", feats_test)
        




def calculate_features(X, users):
    
    feats = pd.DataFrame()
    #text len
    feats['char_count'] = X.map(len)
    #word count
    feats['word_count'] = X.map(lambda x: len(x.split()))
    
    #special features
    #first prons
    reg = r'\bI\b|\bme\b|\bmine\b|\bmy\b|\bmyself\b'
    feats['first_prons'] = X.map(lambda x: len(re.findall(reg, x)))
    # sentiment analysis
    sid = SentimentIntensityAnalyzer()
    feats['sentiment'] = X.map(lambda x: round(sid.polarity_scores(x)['compound'], 2))
    
    nssi_corpus = load_nssi_corpus()
    
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
    
    norm_feats = pd.DataFrame()
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