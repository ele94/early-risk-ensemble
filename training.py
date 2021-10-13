import pandas as pd
import numpy as np
import re
import nltk
import tensorflow.keras.models
from nltk.corpus import stopwords

import numpy
import tensorflow
import sys
import random as rn

from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional, concatenate
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
#from tokenizer import tokenizer as reddit_tokenizer
# my modules
from utils import *



def train(cnn_model=True, maxlen=50000):
    
    logger("Starting training with cnn_model={} and maxlen={}".format(cnn_model, maxlen))
    
    tensorflow.keras.backend.clear_session()
    
    train_feats_new = load_pickle(pickle_path, "feats_train.pkl")
    test_feats_new = load_pickle(pickle_path, "feats_test.pkl")
    X_train = load_pickle(pickle_path, "X_train.pkl")
    X_test = load_pickle(pickle_path, "X_test.pkl")
    y_train = load_pickle(pickle_path, "y_train.pkl")
    y_test = load_pickle(pickle_path, "y_test.pkl")
    
    logger("Generating embeddings")
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    if cnn_model:
        model = define_cnn_model(len(train_feats_new[1,]), vocab_size, embedding_matrix, maxlen)
    else:
        model = define_lstm_model(len(train_feats_new[1,]), vocab_size, embedding_matrix, maxlen)
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    logger("Training")
    history = model.fit([X_train, train_feats_new], y_train, batch_size=2, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    
    logger("Evaluating")
    evaluate_model(model, X_test, test_feats_new, y_test)
    
    logger("Finished training and evaluation")
    
    return model
   


def generate_embeddings_layer(X_train, X_test, maxlen=50000):
    
    tokenizer = Tokenizer(num_words=50000) # 5000
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Padding

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 50000  # podria ser la que quisieramos  # antes tenia 10000, voy a probar con 50000

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    from numpy import array
    from numpy import asarray
    from numpy import zeros

    embeddings_dictionary = dict()
    glove_file = open('/datos/erisk/deep-learning/embeddings/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    
    # Creating an embedding matrix

    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    return X_train, X_test, embedding_matrix, vocab_size


def define_cnn_model(loc_input_len, vocab_size, embedding_matrix, maxlen):
    tensorflow.keras.backend.clear_session()
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,))
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)(nlp_input)
    nlp_out = Conv1D(64, 5, activation='relu')(emb)
    max_pool = GlobalMaxPooling1D()(nlp_out)
    concat = concatenate([max_pool, meta_input])
    classifier = Dense(32, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
        
    return Model(inputs=[nlp_input, meta_input], outputs=[output])


def define_lstm_model(loc_input_len, vocab_size, embedding_matrix, maxlen):
    tensorflow.keras.backend.clear_session()
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,)) 
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)(nlp_input)
    nlp_out = Bidirectional(LSTM(128))(emb)
    concat = concatenate([nlp_out, meta_input])
    classifier = Dense(32, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
    
    return Model(inputs=[nlp_input , meta_input], outputs=[output])

def train_model(model_to_train, X_train, feats_train, y_train):
    history = model_to_train.fit([X_train, feats_train], y_train, batch_size=2, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    return history


def evaluate_model(model_eval, X_test, feats_test, y_test):
    score = model_eval.evaluate([X_test, feats_test], y_test, verbose=1)
    logger("Test Score: {}".format(score[0]))
    logger("Test Accuracy: {}".format(score[1]))

    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    y_pred = model_eval.predict([X_test, feats_test], batch_size=2, verbose=1)
    if y_pred.shape[-1] > 1:
        y_pred_label = y_pred.argmax(axis=-1)
    else:
        print("Entered here")
        y_pred_label = (y_pred > 0.5).astype('int32')

    from sklearn.metrics import classification_report, confusion_matrix

    logger(classification_report(y_test, y_pred_label))
    logger(confusion_matrix(y_test, y_pred_label))
    
    
