import numpy as np

def normal_data(x):
    y = (x - np.mean(x)) / np.std(x)
    return y


