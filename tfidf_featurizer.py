#!/usr/bin/env python
# coding: utf-8
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def combine_features(train_feats, test_feats): # both are a list of the features we want to combine

    train_combined_features = hstack(train_feats)
    test_combined_features = hstack(test_feats)

    print("Is the combined the same from tfidf: {}".format(
        train_feats[0].toarray() == train_combined_features.toarray()))
    
    return train_combined_features, test_combined_features

def tfidf_featurize(X_train, X_test, max_features, only_positives=True, save=False):
    tfidf_vect, xtrain_tfidf, xtest_tfidf = get_features(X_train, X_test, max_features, only_positives=only_positives)
    
    if save:
        save_pickle("pickles", "tfidf_vectorizer.pkl", tfidf_vect)
        save_pickle("pickles", "train_tfidf.pkl", xtrain_tfidf)
        save_pickle("pickles", "test_tfidf.pkl", xtest_tfidf)
    
    return xtrain_tfidf, xtest_tfidf

def get_features(X_train, X_test, max_features, only_positives=True, ngram=False):
    
    if ngram:
        ngram_range = (2, 3)
    else:
        ngram_range = (1, 1)

    train_x = X_train
    test_x = X_test
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                 max_features=max_features, ngram_range=ngram_range)
   
    if only_positives:
        train_x_positives = [text for text, g_truth in zip(train_x['clean_text'], 
                                                           train_x['g_truth']) if g_truth == 1]
        tfidf_vect.fit(train_x_positives)
    else:
        tfidf_vect.fit(train_x['clean_text'])  # aqui pasar solo los positivos???

    xtrain_tfidf = tfidf_vect.transform(train_x["clean_text"])
    xtest_tfidf = tfidf_vect.transform(test_x["clean_text"])
    
    return tfidf_vect, xtrain_tfidf, xtest_tfidf