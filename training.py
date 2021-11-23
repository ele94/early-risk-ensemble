import pandas as pd
import numpy as np
import re
import nltk
import tensorflow.keras.models
from nltk.corpus import stopwords
from tensorflow.keras import backend as K

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
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
#from tokenizer import tokenizer as reddit_tokenizer
# my modules
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import time


def do_train(model_name="cnn_model", maxlen=50000, epochs=1000, early_epochs=10, batch_size=2, shuffle=True, patience=10, model_names=["cnn_model", "lstm_model"], early_stopping=True, validation_split=0.33, 
            feats_train=None, feats_test=None, X_train=None, X_test=None, y_train=None, y_test=None, train_sample_weights=None, save=False, name=None):
    logger("Starting training deep model {}".format(model_name))
    if model_name is "ensemble_model":
        y_pred = do_ensemble(maxlen, batch_size, shuffle, model_names, feats_train, feats_test, X_train, X_test,
                             y_train, y_test, save)
    else:
        y_pred = train(model_name, maxlen, epochs, early_epochs, batch_size, shuffle, patience, early_stopping, 
              validation_split, feats_train, feats_test, X_train, X_test, y_train, y_test, train_sample_weights, save, name)
    return y_pred


def do_ensemble(maxlen=50000, batch_size=2, shuffle=True, model_names=["cnn_model", "lstm_model"], 
               feats_train=None, feats_test=None, X_train=None, X_test=None, y_train=None, y_test=None, save=False):
    
    if feats_train is None:
        logger("Loading variables from memory")
        train_feats_new = load_pickle(pickle_path, "feats_train.pkl")
        test_feats_new = load_pickle(pickle_path, "feats_test.pkl")
        X_train = load_pickle(pickle_path, "X_train.pkl")
        X_test = load_pickle(pickle_path, "X_test.pkl")
        y_train = load_pickle(pickle_path, "y_train.pkl")
        y_test = load_pickle(pickle_path, "y_test.pkl")
        
    else:
        train_feats_new = feats_train
        test_feats_new = feats_test
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    # define input layers and embedding layer
    loc_input_len = train_feats_new.shape[1]
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,))
    model_input = [nlp_input, meta_input]
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    
    # define models
    models = []
    if "cnn_model" in model_names:
        model = define_cnn_model(model_input, emb)
        models.append(model)
    if "lstm_model" in model_names:
        model = define_lstm_model(model_input, emb)
        models.append(model)
    if "lstm_model_32" in model_names:
        model = define_lstm_model_32(model_input, emb)
        models.append(model)
    if "lstm_model_16" in model_names:
        model = define_lstm_model_16(model_input, emb)
        models.append(model)
    
    # reload trained weights
    for i in range(0, len(models)):
        models[i].load_weights("models/{}{}{}{}.hdf5".format(models[i].name, maxlen, batch_size, shuffle))
    
    # reload trained weights
    # cnn_model.load_weights("models/cnn_model" + str(maxlen) + str(batch_size) + str(shuffle) + ".hdf5")
    # lstm_model.load_weights("models/lstm_model" + str(maxlen) + str(batch_size) + str(shuffle) + ".hdf5")
    
    # ensemble
    ensemble_model = define_ensemble_model(models, model_input)
    
    ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # evaluate
    logger("Evaluating ensemble")
    y_pred = evaluate_model(ensemble_model, X_test, test_feats_new, y_test, batch_size)
    
    if save:
        save_pickle(pickle_path, "y_pred.pkl", y_pred)
    return y_pred
    logger("Finished ensemble evaluation")

    
def train_kfolds(model_name="cnn_model", maxlen=50000, batch_size=2, shuffle=True, patience=10, kfolds=10):
    pass
    
    

def train(model_name="cnn_model", maxlen=50000, epochs=100, early_epochs=10, batch_size=2, shuffle=True, patience=10, early_stopping=True, validation_split=0.33, feats_train=None, feats_test=None, X_train=None, X_test=None, y_train=None, y_test=None, train_sample_weights=None, save=False, name=None):
    
    logger("Starting training with model_name={} and maxlen={} and batch size={}".format(model_name, maxlen, batch_size))
    
    tensorflow.keras.backend.clear_session()
    
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      f1_metric,
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    if feats_train is None:
        logger("Loading variables from memory")
        train_feats_new = load_pickle(pickle_path, "feats_train.pkl")
        train_sample_weights = load_pickle("pickles", "X_train_weights.pkl")
        test_feats_new = load_pickle(pickle_path, "feats_test.pkl")
        X_train = load_pickle(pickle_path, "X_train.pkl")
        X_test = load_pickle(pickle_path, "X_test.pkl")
        y_train = load_pickle(pickle_path, "y_train.pkl")
        y_test = load_pickle(pickle_path, "y_test.pkl")
        
    else:
        train_feats_new = feats_train
        test_feats_new = feats_test
    
    logger("Generating embeddings")
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    #X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    #positive_val = len([val for val in y_val if val == 1])
    #negative_val = len([val for val in y_val if val == 0])
    #print("Positive and negative data on validation dataset: {} {}".format(positive_val, negative_val))
    
    # define input layers and embedding layer
    loc_input_len = train_feats_new.shape[1]
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,))
    model_input = [nlp_input, meta_input]
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    
    if model_name is "cnn_model":
        model = define_cnn_model(model_input, emb)
    elif model_name is "lstm_model":
        model = define_lstm_model(model_input, emb)
    elif model_name is "lstm_model_32":
        model = define_lstm_model_32(model_input, emb)
    else:
        model = define_lstm_model_16(model_input, emb)
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    
    logger("Data size: {}".format(X_train.shape[0]))
    
    #early_stopping = tensorflow.keras.callbacks.EarlyStopping(
    #    monitor='val_prc', 
    #    verbose=1,
    #    patience=10,
    #    mode='max',
    #    restore_best_weights=True)
    
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_f1_metric', 
        verbose=1,
        patience=patience,
        mode='max',
        restore_best_weights=True)
    
    filepath = "models/{}{}{}{}{}{}.hdf5".format(model.name, epochs, batch_size, shuffle, patience, name)
    #mc = ModelCheckpoint(filepath, monitor='val_prc', mode='max', verbose=1, save_weights_only=True, save_best_only=True)
    mc = ModelCheckpoint(filepath, monitor='val_f1_metric', mode='max', verbose=0, save_weights_only=True, save_best_only=True)

    #logger("Training with no early stopping")
    #history = model.fit([X_train, train_feats_new], y_train, batch_size=batch_size, steps_per_epoch=20, epochs=early_epochs, verbose=1, validation_split=0.validation_split, shuffle=shuffle, sample_weight=train_sample_weights)
    
    if early_stopping:
        logger("Training with callback")
        history = model.fit([X_train, train_feats_new], y_train, batch_size=batch_size, steps_per_epoch=20, epochs=epochs, verbose=0, validation_split=validation_split, callbacks=[early_stopping, mc], shuffle=shuffle, sample_weight=train_sample_weights)
    else:
        logger("Training with no callback")
        history = model.fit([X_train, train_feats_new], y_train, batch_size=batch_size, steps_per_epoch=20, epochs=epochs, verbose=0, validation_split=validation_split, callbacks=[mc], shuffle=shuffle, sample_weight=train_sample_weights)
    
    logger("Evaluating")
    y_pred = evaluate_model(model, X_test, test_feats_new, y_test, batch_size)
    
    if save:
        save_pickle(pickle_path, "y_pred.pkl", y_pred)
    logger("Finished training and evaluation")
    
    return y_pred



def evaluate_dl_time(model_name="cnn_model", maxlen=50000, epochs=100, early_epochs=10, batch_size=2, shuffle=True, patience=10, early_stopping=True, validation_split=0.33, feats_train=None, feats_test=None, X_train=None, X_test=None, y_train=None, y_test=None, train_sample_weights=None, save=False, name=None):
    
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      f1_metric,
      #keras.metrics.AUC(name='auc'),
      #keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    train_feats_new = feats_train
    test_feats_new = feats_test
    
    X_train, X_test, embedding_matrix, vocab_size = generate_embeddings_layer(X_train, X_test, maxlen)
    
    # define input layers and embedding layer
    loc_input_len = train_feats_new.shape[1]
    meta_input = Input(shape=(loc_input_len,))
    nlp_input = Input(shape=(maxlen,))
    model_input = [nlp_input, meta_input]
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    
    # define models
    if "cnn_model" in model_name:
        model = define_cnn_model(model_input, emb)
    elif "lstm_model" in model_name:
        model = define_lstm_model(model_input, emb)
    elif "lstm_model_32" in model_name:
        model = define_lstm_model_32(model_input, emb)
    else: 
        model = define_lstm_model_16(model_input, emb)
    
    # reload trained weights
    model.load_weights("models/{}{}{}{}{}{}.hdf5".format(model.name, epochs, batch_size, shuffle, patience, name))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
    
    t = time.process_time()
    y_pred = evaluate_model(model, X_test, test_feats_new, y_test, batch_size)
    elapsed_time = time.process_time() - t
    
    return elapsed_time


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


def define_ensemble_model(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    
    model = Model(model_input, y, name='ensemble')
    
    return model


def define_cnn_model(model_input, embedding_layer):
    nlp_input = model_input[0]
    meta_input = model_input[1]
    emb = embedding_layer(nlp_input)
    nlp_out = Conv1D(64, 5, activation='relu')(emb)
    max_pool = GlobalMaxPooling1D()(nlp_out)
    concat = concatenate([max_pool, meta_input])
    classifier = Dense(32, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
        
    return Model(inputs=[nlp_input, meta_input], outputs=output, name="cnn_model")


def define_lstm_model(model_input, embedding_layer):
    nlp_input = model_input[0]
    meta_input = model_input[1]
    emb = embedding_layer(nlp_input)
    nlp_out = Bidirectional(LSTM(128))(emb)
    concat = concatenate([nlp_out, meta_input])
    classifier = Dense(32, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
    
    return Model(inputs=[nlp_input , meta_input], outputs=output, name="lstm_model")

def define_lstm_model_32(model_input, embedding_layer):
    nlp_input = model_input[0]
    meta_input = model_input[1]
    emb = embedding_layer(nlp_input)
    nlp_out = Bidirectional(LSTM(64))(emb)
    concat = concatenate([nlp_out, meta_input])
    classifier = Dense(16, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
    
    return Model(inputs=[nlp_input , meta_input], outputs=output, name="lstm_model_32")

def define_lstm_model_16(model_input, embedding_layer):
    nlp_input = model_input[0]
    meta_input = model_input[1]
    emb = embedding_layer(nlp_input)
    nlp_out = Bidirectional(LSTM(32))(emb)
    concat = concatenate([nlp_out, meta_input])
    classifier = Dense(8, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(classifier)
    
    return Model(inputs=[nlp_input , meta_input], outputs=output, name="lstm_model_16")


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
          
          
def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))