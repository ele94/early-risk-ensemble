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
from bs4 import BeautifulSoup
# my modules
from utils import *

train_users_file = "/datos/erisk/deep-learning/data/erisk2021_training_data/data"
test_users_file = "/datos/erisk/deep-learning/data/erisk2021_test_data/data"

train_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_training_data/golden_truth.txt"
test_g_truth_file = "/datos/erisk/deep-learning/data/erisk2021_test_data/golden_truth.txt"

BPD_file = "/datos/ecampillo/jupyter/dl-notebooks/newensemble/early-risk-ensemble/new_data/BPD"
BPD_file_2 = "/datos/ecampillo/jupyter/dl-notebooks/newensemble/early-risk-ensemble/new_data/BorderlinePDiagnosed"
BPD_file_3 = "/datos/ecampillo/jupyter/dl-notebooks/newensemble/early-risk-ensemble/All_Users"

new_data_filepaths = [BPD_file, BPD_file_2, BPD_file_3]

# preprocessing the text
import redditcleaner

R_tokenizer = reddit_tokenizer.TweetTokenizer(preserve_case=False, preserve_url=False,
                                    regularize=True, preserve_emoji=False, preserve_hashes=False,
                                   preserve_handles=False)


# public callable method, the other ones are helpers
def preprocess():
    
    logger("Starting preprocessing")
    seed(42)
    tensorflow.random.set_seed(42) 
    logger("Initialized numpy random and tensorflow random seed at 42")
    
    train_g_truth = load_golden_truth(train_g_truth_file)
    test_g_truth = load_golden_truth(test_g_truth_file, test_collection=True)

    logger("Loading user data")
    train_users = load_user_data(train_users_file, train_g_truth)
    test_users = load_user_data(test_users_file, test_g_truth, test_collection=True)
    
    logger("Loading new user data")
    train_users_new = load_new_data()
    
    logger("Preprocessing user data")
    train_users = preprocess_data(train_users)
    test_users = preprocess_data(test_users)
    train_users_new = preprocess_data(train_users_new)
    
    logger("Saving preprocessed data")
    save_pickle("pickles", "raw_train_users.pkl", train_users)
    save_pickle("pickles", "raw_test_users.pkl", test_users)
    save_pickle("pickles", "raw_train_users_new.pkl", train_users_new)
    
    logger("Finished preprocessing")
    

def load_new_data():
    for new_data_filepath in new_data_filepaths:
        for filename in os.listdir(new_data_filepath):
            if ".txt" in filename:
                filepath = os.path.join(new_data_filepath, filename)
                fix_xml_tags(filepath)

    new_users = {}
    for new_data_filepath in new_data_filepaths:
        g_file = make_g_file(new_data_filepath)
        new_users.update(load_new_user_data(new_data_filepath, g_file))
    
    return new_users

    
def preprocess_data(users):
    
    preproc_users = {}
    logger("Preprocessing {} users".format(len(users.keys())))
    for user, writings in users.items():
        preproc_writings = []
        for writing in writings:
            writing["clean_text"] = preprocess_text_v2(preprocess_text(writing["text"]))
            writing["tokens"] = tokenize_text(writing["clean_text"])
            writing["pos_tags"] = pos_tag_text(writing["tokens"])
            writing["stems"] = stemmize_text(writing["tokens"])
            preproc_writings.append(writing)

        preproc_users[user] = preproc_writings
        write_loading_bar(len(users.keys()), len(preproc_users.keys()))
    
    return preproc_users
        
        
def calculate_preproc(df):
    df = pd.DataFrame(df)
    df["clean_text"] = df["text"].apply(preprocess_text)
    df["clean_text"] = df["clean_text"].apply(preprocess_text_v2)
    logger("Finished cleaning text")
    df["tokens"] = df["clean_text"].apply(tokenize_text)
    logger("Finished calculating tokens")
    df["pos_tags"] = df["tokens"].apply(pos_tag_text)
    logger("Finished obtaining pos tags")
    df["stems"] = df["tokens"].apply(stemmize_text)
    logger("Finished calculating stems")
    return df


def fix_xml_tags(path):
    # Read in the file
    with open(path, 'r', encoding="utf-8") as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('\t', '    ')
    filedata = filedata.replace('    TEXT>', '    <TEXT>')

        # Write the file out again
    with open(path, 'w', encoding="utf-8") as file:
        file.write(filedata)
        
def load_new_user_data(path, g_truth, test_collection=False):
    users = {}
    for filename in os.listdir(path):
        user_writings = []
        old_user, file_extension = os.path.splitext(filename)
        user = str(old_user)
        
        filepath_name = os.path.join(path, filename)
        with open(filepath_name, encoding="utf-8") as fp:
            soup = BeautifulSoup(fp, 'xml')
    
        writings = soup.find_all('TEXT')
        logger("Length of user writings: {}".format(len(writings)))
        for writing in writings:
            user_writing = {"text": writing.text, "user": user, "g_truth": g_truth[user]}
            user_writings.append(user_writing)
        users[user] = user_writings
        
    return users

def make_g_file(path):
    g_file = {}
    for filename in os.listdir(path):
        user_id, file_extension = os.path.splitext(filename)
        g_file[user_id] = 1
    return g_file
    

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
    users = {}
    #user_writings = []
    for filename in os.listdir(path):
        old_user, file_extension = os.path.splitext(filename)
        
        if test_collection:
            user = "test"+str(old_user)
        else:
            user = str(old_user)
        tree = ET.parse(os.path.join(path, filename))
        root = tree.getroot()
        user_writings = []
        
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
        users[user] = user_writings
        
    return users

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