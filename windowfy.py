# -----------------------------------------------------------
# Implements windowfy functionality from eRisk model
#
# email ecampillo@lsi.uned.es
# -----------------------------------------------------------

from utils import *
from preprocessing import preprocess
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

train_users_file = "raw_train_users.pkl"
test_users_file = "raw_test_users.pkl"
pickles_dir = "pickles"
new_train_users_file = "raw_train_users_new.pkl"

def windowfy(window_size=10, sample_weights_size=10, max_size=10000, is_oversample=False, include_new_data=False):
    """Obtains the text windows from preprocessed data stored in pickles/raw_train_users.pkl and pickles/raw_test_users.pkl
    
    :param window_size: the number of messages to join in a window
    :param sample_weights_size: the size of the sample weights
    :param max_size: the maximum number of messages to take from each user
    :param is_oversample: indicates if training data should be oversampled
    :param include_new_data: indicates if new data from BPD forums should be loaded too
    
    """
    
    train_users = load_pickle(pickles_dir, train_users_file)
    test_users = load_pickle(pickles_dir, test_users_file)
    if include_new_data:
        train_users.update(load_pickle(pickles_dir, new_train_users_file))
    
    # sequence numbers were not loaded during preprocessing, so we calculate them now
    train_users = assign_sequence_numbers(train_users)    
    test_users = assign_sequence_numbers(test_users)

    logger("Windowfying training users")
    train_window = windowfy_sliding(train_users, window_size, max_size)
    logger("\nWindowfying test users")
    test_window = windowfy_sliding(test_users, window_size, max_size)
    
    train_window_frame = pd.DataFrame(train_window)
    test_window_frame = pd.DataFrame(test_window)
    
    X_train = train_window_frame["clean_text"] 
    X_test = test_window_frame["clean_text"]
    y_train = np.array(train_window_frame["g_truth"])
    y_test = np.array(test_window_frame["g_truth"])
    
    if is_oversample:
        logger("\nOversampling train users")
        train_window_frame, y_train = oversample(train_window_frame, y_train)
        positive_messages = len([message for message in y_train if message == 1])
        negative_messages = len([message for message in y_train if message == 0])
        print("After oversample: positive messages: {}, negative messages: {}".format(positive_messages, negative_messages))
        X_train = train_window_frame["clean_text"]
    
    # we get the train window sample weights
    train_samples = assign_sequence_weights(train_window_frame, max_size, sample_weights_size)
    train_samples = np.array(train_samples)
        
    logger("Data size: {}".format(X_train.shape[0]))
    
    logger("\nFinished windowfying")
    return train_window_frame, y_train, test_window_frame, y_test, train_samples, X_train, X_test


#----------------------------------------------------------------
# Helper methods, not to be called from outside of this module
#----------------------------------------------------------------

def oversample(train_users, y_train):
    """ Obtains oversampled data from train_users and updates y_train accordingly """
    
    ros = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_users, y_train)
    
    return X_resampled, y_resampled


def windowfy_sliding(users, window_size, max_size):
    """ Joins users text comments into windows of size window_size up to a maximum length of max_size """
    
    users_windows = []
    count_users = 0 # used to print the progress bar
    
    for user, user_writings in users.items():
        writings = user_writings.copy()
        writings = writings[:max_size] # cut the writings to the maximum length
        for i in range(0, len(writings)):
            window = writings[i:i+window_size]
            joined_window = join_window_elements(window) # join the texts from the window
            users_windows.append(joined_window)
            # si el numero de mensajes es menor que el tamaño de la ventana 
            # es que hemos llegado al final, asi que salimos
            if len(window) < window_size:
                break
                
        count_users += 1
        write_loading_bar(len(users.keys()), count_users) # prints a pretty progress bar
        
    return users_windows    


def join_window_elements(window: list) -> dict:
    """ Joins a number of messages in a list to a single message with appended text, a single g_truth, a list of dates,
    the sequence value of the last message, etc.
    """
    
    joint_window = {}
    flatten = lambda l: [item for sublist in l for item in sublist]

    for key in window[0].keys():
        key_list = [message[key] for message in window]
        if type(key_list[0]) is list:
            joint_window[key] = flatten(key_list)
        elif key == 'user':
            joint_window[key] = key_list[0]
        elif key == 'g_truth':
            joint_window[key] = key_list[0]
        elif key == 'date':
            joint_window[key] = key_list
        elif key == 'sequence':
            joint_window[key] = key_list[-1]
        else:
            joint_window[key] = ' .'.join(key_list)

    return joint_window

         
def assign_sequence_weights(train_wf, max_range=10000, samples_window=10):
    """ obtains sample weights from compressed train_wf dataframe taking into account the sequence number 
    of the window    
    """ 
    
    sample_weights = get_sequence_weights(samples_window=samples_window, max_range=max_range)
    train_window_weights = [sample_weights[g_truth][sequence] for sequence, g_truth in zip(train_wf['sequence'], train_wf['g_truth'])]
        
    return train_window_weights


def get_sequence_weights(samples_window=10, max_range=10000):
    """ Creates a dictionary of sample weight values in order with sample weights for 0 and 1 """
    
    sample_weights = {0: [], 1: []}
    max_size = samples_window
    positive_sample_weights = [x / (1.0 * max_size) for x in range((1 * max_size), (2 * max_size), 1)]
    positive_sample_weights.extend(np.ones(max_range))
    positive_sample_weights = positive_sample_weights[:max_range]
    negative_sample_weights = np.ones(max_range)
    sample_weights[0] = negative_sample_weights
    sample_weights[1] = positive_sample_weights
    return sample_weights


def assign_sequence_numbers(train_users):
    """ Adds a field called sequence to each user message with the order of the message """
    
    train_users_iter = train_users.copy()
    for user, writings in train_users_iter.items():
        for index, _ in enumerate(writings):
            train_users[user][index]["sequence"] = index
            
        
    return train_users