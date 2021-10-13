from sklearn import naive_bayes, ensemble
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from numpy import array
from numpy.random import seed
from sklearn.model_selection import train_test_split
from tokenizer import tokenizer as reddit_tokenizer
#from redditscore.tokenizer import CrazyTokenizer #https://github.com/crazyfrogspb/RedditScore
import copy
import numpy as np
import re
import sys
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
import sklearn.metrics as metrics
import pickle
# my modules
from utils import *

train_users_file = "/datos/erisk/deep-learning/data/erisk2021_training_data/data"
test_users_file = "/datos/erisk/deep-learning/data/erisk2021_test_data/data"

train_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_training_data/golden_truth.txt"
test_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_test_data/golden_truth.txt"


# preprocessing the text
import redditcleaner

R_tokenizer = reddit_tokenizer.TweetTokenizer(preserve_case=False, preserve_url=False,
                                    regularize=True, preserve_emoji=False, preserve_hashes=False,
                                   preserve_handles=False)


# public callable method, the other ones are helpers
def preprocess(is_oversample=True, join_data=True):
    
    logger("Starting preprocessing with oversample: {} and join_data: {}".format(is_oversample, join_data))
    seed(42)
    tensorflow.random.set_seed(42) 
    logger("Initialized numpy random and tensorflow random seed at 42")
    
    train_g_truth = load_golden_truth(train_g_truth_file)
    test_g_truth = load_golden_truth(test_g_truth_file, test_collection=True)

    if join_data:
        logger("Loading joined user data")
        train_users = load_joined_user_data(train_users_file, train_g_truth)
        test_users = load_joined_user_data(test_users_file, test_g_truth, test_collection=True)
    else:
        logger("Loading user data")
        train_users = load_user_data(train_users_file, train_g_truth)
        test_users = load_user_data(test_users_file, test_g_truth, test_collection=True)
        
    train_users = pd.DataFrame(train_users)
    test_users = pd.DataFrame(test_users)
    
    train_users["clean_text"] = train_users["text"].apply(preprocess_text)
    test_users["clean_text"] = test_users["text"].apply(preprocess_text)
    
    train_users["clean_text"] = train_users["clean_text"].apply(preprocess_text_v2)
    test_users["clean_text"] = test_users["clean_text"].apply(preprocess_text_v2)
    
    train_users["tokens"] = train_users["clean_text"].apply(tokenize_text)
    test_users["tokens"] = test_users["clean_text"].apply(tokenize_text)
    train_users["pos_tags"] = train_users["tokens"].apply(pos_tag_text)
    test_users["pos_tags"] = test_users["tokens"].apply(pos_tag_text)
    train_users["stems"] = train_users["tokens"].apply(stemmize_text)
    test_users["stems"] = test_users["tokens"].apply(stemmize_text)
    
    X_train = train_users["clean_text"] 
    X_test = test_users["clean_text"]
    
    y_train = np.array(train_users["g_truth"])
    y_test = np.array(test_users["g_truth"])
    
    if is_oversample:
        train_users, y_train = oversample(train_users, y_train)
        X_train = train_users["clean_text"]
        
    save_pickle("pickles", "X_train.pkl", X_train)
    save_pickle("pickles", "X_test.pkl", X_test)
    save_pickle("pickles", "train_users.pkl", train_users)
    save_pickle("pickles", "test_users.pkl", test_users)
    save_pickle("pickles", "y_train.pkl", y_train)
    save_pickle("pickles", "y_test.pkl", y_test)
    
    logger("Finished preprocessing")

    

import sklearn
from imblearn.over_sampling import RandomOverSampler
    
def oversample(train_users, y_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_users, y_train)
    
    return X_resampled, y_resampled


def preprocess_text(sen):
    tokens = R_tokenizer.tokenize(sen)
    sentence = " ".join(tokens)
    return sentence

def preprocess_text_v2(sen):
    # Cleaning reddit text
    sentence = clean_reddit_text(sen)
    
    # Removing html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    # sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Lower case
    sentence = sentence.lower()

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def clean_reddit_text(text):
    return redditcleaner.clean(text)

# new

def tokenize_text(text):
    text = text.lower()
    text = remove_stopwords(text)
    text = word_tokenize(text)
    return text

# text tiene que venir en tokens
def pos_tag_text(text):
    text = nltk.pos_tag(text)
    return text

from nltk.corpus import stopwords
import re
def remove_stopwords(text):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', text)
    return text

from nltk.stem import PorterStemmer
def stemmize_text(text):
    ps = PorterStemmer()
    stems = [ps.stem(w) for w in text]
    return stems

import os
import xml.etree.ElementTree as ET

def load_golden_truth(g_path, test_collection=False):
    g_truth = {line.split()[0]: int(line.split()[1]) for line in open(g_path)}
    if test_collection:
        new_g_truth = {}
        for user, truth in g_truth.items():
            new_g_truth["test"+user] = truth
    else:
        new_g_truth = g_truth.copy()
    return new_g_truth

def load_user_data(path, g_truth, test_collection=False):
    #users = {}
    user_writings = []
    for filename in os.listdir(path):
        old_user, file_extension = os.path.splitext(filename)
        
        if test_collection:
            user = "test"+str(old_user)
        else:
            user = str(old_user)
        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()
        #user_writings = []
        
        for writing in root.findall('WRITING'):
            title, text, date = "", "", ""
            if writing.find('TITLE') is not None:
                title = writing.find('TITLE').text
                if title is None:
                    title = ""
            if writing.find('TEXT') is not None:
                text = writing.find('TEXT').text
                if text is None:
                    text = ""
                    
            if len(title) > 0:
                user_writing = {"text": title + ". " + text, "user": user, "g_truth": g_truth[user]}
            else:
                user_writing = {"text": text, "user": user, "g_truth": g_truth[user]}
            user_writings.append(user_writing)
        #users[user] = user_writings
        
    return user_writings

def load_joined_user_data(path, g_truth, test_collection=False):
    user_writings = []
    for filename in os.listdir(path):
        old_user, file_extension = os.path.splitext(filename)
        
        if test_collection:
            user = "test"+str(old_user)
        else:
            user = str(old_user)
        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()
        writings = []
        
        for writing in root.findall('WRITING'):
            title, text, date = "", "", ""
            if writing.find('TITLE') is not None:
                title = writing.find('TITLE').text
                if title is None:
                    title = ""
            if writing.find('TEXT') is not None:
                text = writing.find('TEXT').text
                if text is None:
                    text = ""
                    
            if len(title) > 0:
                writings.append(title + ". " + text)
            else:
                writings.append(text)
        writings = ". ".join(writings)
        user_writings.append({"text": writings, "user": user, "g_truth": g_truth[user]})
        #users[user] = user_writings
        
    return user_writings