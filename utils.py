# Python Module example
import sys
import pickle
import os

# Funciones utiles

pickle_path = "/datos/ecampillo/jupyter/dl-notebooks/newensemble/early-risk-ensemble/pickles"

def set_pickle_path(pickle_path):
    pickle_path = pickle_path

def logger(message, debug_file="log.txt"):
    print(message)
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(debug_file, 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(message)
        sys.stdout = original_stdout # Reset the standard output to its original value
        
def save_pickle(filepath, filename, data):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    file = os.path.join(filepath, filename)
    with open(file, 'wb') as data_file:
        pickle.dump(data, data_file)
        
def load_pickle(filepath, filename):
    file = os.path.join(filepath, filename)
    with open(file, 'rb') as data_file:
        data = pickle.load(data_file)
    return data