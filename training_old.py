import pandas as pd
import numpy as np
import re
import nltk
import tensorflow.keras.models
from nltk.corpus import stopwords

import numpy
import tensorflow
from tensorflow import keras
import sys
import random as rn

from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Dropout, Dense, Average
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional, concatenate
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
#from tokenizer import tokenizer as reddit_tokenizer
# my modules
from utils import *


def do_ensemble(maxlen=50000, batch_size=2, shuffle=True):
    
    train_feats_new = load_pickle(pickle_path, "feats_train.pkl")
    test_feats_new = load_pickle(pickle_path, "feats_test.pkl")
    X_train = load_pickle(pickle_path, "X_train.pkl")
    X_test = load_pickle(pickle_path, "X_test.pkl")
    y_train = load_pickle(pickle_path, "y_train.pkl")
    y_test = load_pickle(pickle_path, "y_test.pkl")
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    cnn_model = load_model("models/cnn_model" + str(maxlen) + str(batch_size) + str(shuffle) + ".h5")
    lstm_model = load_model("models/lstm_model" + str(maxlen) + str(batch_size) + str(shuffle) + ".h5")
    models = [cnn_model, lstm_model]
    
    loc_input_len = train_feats_new.shape[1]
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,))
    model_inputs = [nlp_input, meta_input]
    
    ensemble_model = ensemble(models, model_inputs)
    
    logger("Evaluating")
    y_pred = evaluate_model(ensemble_model, X_test, test_feats_new, y_test, batch_size)
    
    save_pickle(pickle_path, "y_pred.pkl", y_pred)
    logger("Finished ensemble evaluation")


def train(cnn_model=True, maxlen=50000, batch_size=2, shuffle=True):
    
    logger("Starting training with cnn_model={} and maxlen={} and batch size={}".format(cnn_model, maxlen, batch_size))
    
    tensorflow.keras.backend.clear_session()
    
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    
    train_feats_new = load_pickle(pickle_path, "feats_train.pkl")
    test_feats_new = load_pickle(pickle_path, "feats_test.pkl")
    X_train = load_pickle(pickle_path, "X_train.pkl")
    X_test = load_pickle(pickle_path, "X_test.pkl")
    y_train = load_pickle(pickle_path, "y_train.pkl")
    y_test = load_pickle(pickle_path, "y_test.pkl")
    
    logger("Generating embeddings")
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    if cnn_model:
        model = define_cnn_model(train_feats_new.shape[1], vocab_size, embedding_matrix, maxlen)
    else:
        model = define_lstm_model(train_feats_new.shape[1], vocab_size, embedding_matrix, maxlen)
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    
    logger("Data size: {}".format(X_train.shape[0]))
    
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    
    filepath = 'models/' + model.name + str(maxlen) + str(batch_size) + str(shuffle) + '.h5'
    mc = ModelCheckpoint(filepath, monitor='val_prc', mode='max', verbose=1, save_best_only=True)


    logger("Training")
    history = model.fit([X_train, train_feats_new], y_train, batch_size=batch_size, steps_per_epoch=20, epochs=100, verbose=1, validation_split=0.2, callbacks=[early_stopping, mc], shuffle=shuffle)
    
    logger("Evaluating")
    y_pred = evaluate_model(model, X_test, test_feats_new, y_test, batch_size)
    
    save_pickle(pickle_path, "y_pred.pkl", y_pred)
    logger("Finished training and evaluation")
    
    return model
   

def ensemble(models, model_input):
    
    outputs = [model.outputs[0] for model in models]
    for output in outputs:
        print(output)
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    print(model.summary())
    
    return model
    

def generate_embeddings_layer(X_train, X_test, maxlen=50000):
    
    tokenizer = Tokenizer(num_words=50000) # 5000
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Padding

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    # maxlen podria ser la que quisieramos  # antes tenia 10000, voy a probar con 50000

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
        
    return Model(inputs=[nlp_input, meta_input], outputs=output, name="cnn_model")


def define_lstm_model(loc_input_len, vocab_size, embedding_matrix, maxlen):
    tensorflow.keras.backend.clear_session()
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,)) 
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)(nlp_input)
    nlp_out = Bidirectional(LSTM(128))(emb)
    concat = concatenate([nlp_out, meta_input])
    classifier = Dense(32, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
    
    return Model(inputs=[nlp_input , meta_input], outputs=output, name="lstm_model")

def train_model(model_to_train, X_train, feats_train, y_train):
    history = model_to_train.fit([X_train, feats_train], y_train, batch_size=2, epochs=10, verbose=1, validation_split=0.2, shuffle=True)
    return history


def evaluate_model(model_eval, X_test, feats_test, y_test, batch_size):
    score = model_eval.evaluate([X_test, feats_test], y_test, verbose=1)
    logger("Test Score: {}".format(score[0]))
    logger("Test Accuracy: {}".format(score[1]))

    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    y_pred = model_eval.predict([X_test, feats_test], batch_size=batch_size, verbose=1)
    if y_pred.shape[-1] > 1:
        y_pred_label = y_pred.argmax(axis=-1)
    else:
        logger("Entered here")
        y_pred_label = (y_pred > 0.5).astype('int32')

    from sklearn.metrics import classification_report, confusion_matrix

    logger(classification_report(y_test, y_pred_label))
    logger(confusion_matrix(y_test, y_pred_label))
    
    return y_pred_label
    
    
