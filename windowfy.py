from utils import *
from preprocessing import preprocess
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def windowfy(window_size=10, is_oversample=False):
    
    train_users = load_pickle("pickles", "raw_train_users.pkl")
    test_users = load_pickle("pickles", "raw_test_users.pkl")

    logger("Windowfying training users")
    train_window = windowfy_sliding_training(train_users, window_size)
    logger("\nWindowfying test users")
    test_window = windowfy_sliding_testing(test_users, window_size)
    
    train_window_frame = pd.DataFrame(train_window)
    test_window_frame = pd.DataFrame(test_window)
    
    X_train = train_window_frame["clean_text"] 
    X_test = test_window_frame["clean_text"]
    y_train = np.array(train_window_frame["g_truth"])
    y_test = np.array(test_window_frame["g_truth"])
    
    if is_oversample:
        logger("\nOversampling train users")
        train_window_frame, y_train = oversample(train_window_frame, y_train)
        X_train = train_window_frame["clean_text"]
        
    logger("Data size: {}".format(X_train.shape[0]))
    
    save_pickle("pickles", "X_train.pkl", X_train)
    save_pickle("pickles", "X_test.pkl", X_test)
    save_pickle("pickles", "y_train.pkl", y_train)
    save_pickle("pickles", "y_test.pkl", y_test)
    save_pickle("pickles", "train_users.pkl", train_window_frame)
    save_pickle("pickles", "test_users.pkl", test_window_frame)
    
    logger("\nFinished windowfying")

# methods

def oversample(train_users, y_train):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_users, y_train)
    
    return X_resampled, y_resampled

def windowfy_sliding_training(users, window_size, pos_param_range_max=-1, neg_param_range_max=-1):
    users_windows = []
    count_users = 0
    for user, user_writings in users.items():
        
        if user_writings[0]["g_truth"] == 1:
            param_range_max = pos_param_range_max
        else:
            param_range_max = neg_param_range_max
        if param_range_max < 0 or param_range_max > len(user_writings):
            range_max = len(user_writings)
            writings = user_writings.copy()
        else:
            range_max = param_range_max
            writings = user_writings.copy()[:range_max]
        for i in range(0, range_max):  # TODO parametrizar esto
            #if i < window_size and len(writings) > (i + 1):
            #    window = writings[:i + 1]  # rellenamos mientras "no nos llegan los demas mensajes"
            if len(writings) < (i + window_size):
                window = writings[i:range_max]  # TODO comprobar este range_max
                continue
            else:
                window = writings[i:i + window_size]

            if len(window) == 0:
                pass
                #print("Window: {}, i: {}, len(writings): {}".format(window, i, len(writings)))

            joined_window = join_window_elements(window)
            if len(joined_window["text"]) >= 10: 
                users_windows.append(joined_window)
            
        count_users += 1
        write_loading_bar(len(users.keys()), count_users)

    return users_windows


def windowfy_sliding_testing(users, window_size, param_range_max=-1):
    users_windows = []
    count_users = 0
    for user, writings in users.items():
        if param_range_max < 0 or param_range_max > len(writings):
            range_max = len(writings)
        else:
            range_max = param_range_max
        for i in range(0, range_max):  # TODO parametrizar esto
            if i < window_size and len(writings) > (i+1):
                window = writings[:i+1] # rellenamos mientras "no nos llegan los demas mensajes" # todo cambiar esto
            elif len(writings) < (i + window_size):
                #window = writings[i:range_max]  # TODO comprobar este range_max
                window = []
            else:
                window = writings[i:i + window_size]

            if len(window) == 0:
                pass
                #print("Window: {}, i: {}, len(writings): {}".format(window, i, len(writings)))
            else:
                joined_window = join_window_elements(window)
                users_windows.append(joined_window)
                
        count_users += 1
        write_loading_bar(len(users.keys()), count_users)

    return users_windows


def join_all_elements(users):
    joined_writings = []
    for user, writings in users.items():
        joined_writings.append(join_window_elements(writings))

    return joined_writings


def join_window_elements(window: list) -> dict:
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