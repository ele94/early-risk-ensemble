from utils import *
from preprocessing import preprocess
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def windowfy(window_size=10, sample_weights_size=10, max_size=10000, is_oversample=False, include_new_data=False, save=False):
    
    train_users = load_pickle("pickles", "raw_train_users.pkl")
    test_users = load_pickle("pickles", "raw_test_users.pkl")
    if include_new_data:
        train_users.update(load_pickle("pickles", "raw_train_users_new.pkl"))
        
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
    
    if save:
        logger("Saving variables to memory")
        save_pickle("pickles", "X_train.pkl", X_train)
        save_pickle("pickles", "X_train_weights.pkl", train_samples)
        save_pickle("pickles", "X_test.pkl", X_test)
        save_pickle("pickles", "y_train.pkl", y_train)
        save_pickle("pickles", "y_test.pkl", y_test)
        save_pickle("pickles", "train_users.pkl", train_window_frame)
        save_pickle("pickles", "test_users.pkl", test_window_frame)
    
    logger("\nFinished windowfying")
    return train_window_frame, y_train, test_window_frame, y_test, train_samples, X_train, X_test

# methods

def oversample(train_users, y_train):
    ros = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_users, y_train)
    
    return X_resampled, y_resampled


def windowfy_sliding(users, window_size, max_size):
    users_windows = []
    count_users = 0
    for user, user_writings in users.items():
        writings = user_writings.copy()
        writings = writings[:max_size]
        for i in range(0, len(writings)):
            window = writings[i:i+window_size]
            joined_window = join_window_elements(window)
            users_windows.append(joined_window)
            # si el numero de mensajes es menor que el tamaño de la ventana 
            # es que hemos llegado al final, asi que salimos
            if len(window) < window_size:
                break
                
        count_users += 1
        write_loading_bar(len(users.keys()), count_users)
        
    return users_windows    
        
        
def windowfy_sliding_weights(sample_weights, window_size, max_size):
    weights_windows = []
    count_users = 0
    for user, user_sample_weights in users.items():
        user_samples = user_sample_weights.copy()
        user_samples = user_samples[:max_size]
        for i in range(0, len(user_samples)):
            window = user_samples[i+window_size] # we select the weight of the last message of the window
            weights_windows.append(weights_window)
            # si el numero de mensajes es menor que el tamaño de la ventana 
            # es que hemos llegado al final, asi que salimos
            if len(window) < window_size:
                break
                
        count_users += 1
        write_loading_bar(len(users.keys()), count_users)
        
    return users_windows 


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
                break
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
                break
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


# obtains sample weights for all the messages of the users before any kind of cropping or windowfying takes place
# train_users: user data that we want to obtain sample weights for
# sample_window: last message above 1 for positive messages
def get_sample_weights(train_users, samples_window=10):
    train_users_samples = {}
    max_size = samples_window
    
    for user, user_writings in train_users.items():
        if user_writings[0]["g_truth"] == 1:
            # generates array of sample weights from 2 to 1 in max_size steps
            sample_weights = [x / (1.0 * max_size) for x in range((1 * max_size), (2 * max_size), 1)]
            # we populate the rest of the array so it is the same size as the user_writings
            sample_weights.append(np.ones(len(user_writings)))
            sample_weights = sample_weights[:user_writings]
        else:
            # negative users always have smaple weight 1
            sample_weights = np.ones(len(range_max))
            
        train_users_samples[user] = sample_weights
    
    return train_users_samples

def assign_sequence_numbers(train_users):
    
    train_users_iter = train_users.copy()
    for user, writings in train_users_iter.items():
        for index, _ in enumerate(writings):
            train_users[user][index]["sequence"] = index
            
        
    return train_users
            
    
    
        
# obtains sample weights for blah       
def get_sequence_weights(samples_window=10, max_range=10000):
    sample_weights = {0: [], 1: []}
    max_size = samples_window
    positive_sample_weights = [x / (1.0 * max_size) for x in range((1 * max_size), (2 * max_size), 1)]
    positive_sample_weights.extend(np.ones(max_range))
    positive_sample_weights = positive_sample_weights[:max_range]
    negative_sample_weights = np.ones(max_range)
    sample_weights[0] = negative_sample_weights
    sample_weights[1] = positive_sample_weights
    return sample_weights

# obtains sample weights from compressed train_window dataframe taking into account the sequence number of the window        
def assign_sequence_weights(train_wf, max_range=10000, samples_window=10):
    sample_weights = get_sequence_weights(samples_window=samples_window, max_range=max_range)
    train_window_weights = [sample_weights[g_truth][sequence] for sequence, g_truth in zip(train_wf['sequence'], train_wf['g_truth'])]
        
    return train_window_weights
        
    