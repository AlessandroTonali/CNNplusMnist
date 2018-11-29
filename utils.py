import numpy as np

def gradient_multiplier(out_layer, sigma):
    a = np.array(out_layer)
    b = np.array(sigma)
    return np.tensordot(sigma,out_layer, 0)