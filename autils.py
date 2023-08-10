import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def load_data():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the new path by concatenating the script directory and the relative path
    relative_path_X = os.path.join(script_directory, 'data/X.npy')
    relative_path_y = os.path.join(script_directory, 'data/y.npy')

    X = np.load(relative_path_X)
    y = np.load(relative_path_y)
    X = X[0:5000]
    y = y[0:5000]
    return X, y