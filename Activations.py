import numpy as np


def prime_function(activation):
    if activation == sigmoid:
        return sigmoid_prime
    if activation == relu:
        return relu_prime
    if activation == leaky_relu:
        return leaky_relu_prime
    if activation == tanh:
        return tanh_prime

def sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    a = np.array([max(0,i) for i in x])
    return a

def relu_prime(x):
    return np.array([(0 if i <= 0 else 1) for i in x])

def leaky_relu(x):
    a = np.array([(0.01*i*(i < 0) + i*(i >0)) for i in x])
    return a

def leaky_relu_prime(x):
    return np.array([0.1 if i <= 0 else 1 for i in x])

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.power(np.tanh(x),2)












