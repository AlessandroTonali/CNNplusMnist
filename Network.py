import numpy as np
from numpy.random import randn
from Activations import sigmoid, prime_function, relu, leaky_relu
from LossFunctions import dSquaredLoss
from collections import deque
from utils import gradient_multiplier
from database_loader import load_data
from time import time
from random import shuffle
from copy import deepcopy



class Network:
    def __init__(self, sizes, activation_funcs):
        self.num_layers = len(sizes)
        self.activations = activation_funcs
        self.sizes = sizes
        self.biases = [randn(y, 1) for y in sizes[1:]]
        self.weights = [randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]




    def inference(self, x, cut_off=None):  # the cutoff can be set to get the output of some hidden layer
        if cut_off == None or cut_off > self.num_layers:  # if cut_off not set or invalid put the standard one
            output = self.num_layers
        else:
            output = cut_off

        out_layer = np.array(x)


        for l in range(1, output): # This cycle iterates on the layers
            activation = self.activations[l]
            out_layer = np.matmul(self.weights[l-1], out_layer)
            out_layer = out_layer + np.transpose(self.biases[l-1])[0]
            out_layer = activation(out_layer)
        return out_layer


    def training(self, trainData, T, n, alpha, lmbda, validation, patience=10):
        """
        trains the ANN with training dataset using stochastic gradient descent
        :param trainData: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param T: total number of iteration
        :param n: size of mini-batches
        :param alpha: learning rate
        """
        patience_count = 10
        max_acc = 0
        for i in range(0, T):
            shuffle(trainData)
            batch_num = 0
            while (batch_num+1) * n <= len(trainData):
                self.updateWeights(trainData[batch_num*n:((batch_num+1)*n)], alpha, lmbda)
                batch_num = batch_num + 1
            if batch_num*n != len(trainData):
                self.updateWeights(trainData[batch_num * n:(len(trainData))], alpha, lmbda)
            val_acc = self.evaluate(validation)
            if val_acc > max_acc:
                max_acc = val_acc
                weights = deepcopy(self.weights)
                patience_count = patience
            else:
                patience_count -=1
                if patience_count == 0:
                    assert isinstance(weights, list)
                    self.weights = weights
                    return








    def updateWeights(self, batch, alpha, lmbda=0):
        """
        called by 'training', update the weights and biases of the ANN
        :param batch: mini-batch, a list of pair (x, y)
        :param alpha: learning rate
        """
        i = 0
        couple = batch[i]
        
        grad = self.backprop(couple[0], couple[1])

        res_bias = grad[1]

        res_weight = grad[0]

        i += 1
        while i < len(batch):
            couple = batch[i]
            grad = self.backprop(couple[0], couple[1])
            bias_grad = grad[1]

            weight_grad = grad[0]
            j = 0
            while j < len(res_bias):
                res_bias[j] = bias_grad[j] + res_bias[j]
                j = j + 1
            j = 0
            while j < len(res_weight):
                res_weight[j] = weight_grad[j] + res_weight[j]
                j = j + 1
            j = 0

            i += 1
        z = 1 - alpha* lmbda/ len(batch)
        j = 0
        while j < len(self.weights):
            self.weights[j] = (1 - alpha * lmbda) * self.weights[j] - alpha/len(batch) * res_weight[j]
            j += 1
        j = 0

        while j < len(self.biases):
            self.biases[j] = (1 - alpha * lmbda) * self.biases[j] - alpha / len(batch) * res_bias[j]
            j += 1








    def backprop(self, x, y):

        """
        called by 'updateWeights'
        :param: (x, y): a tuple of batch in 'updateWeights'
        :return: a tuple (nablaW, nablaB) representing the gradient of the empirical risk for an instance x, y
                nablaW and nablaB follow the same structure as self.weights and self.biases

        """
        a_list = []
        z_list =[]

        bias_grad = deque()
        weight_grad = deque()
        if isinstance(x, int):
            a_list.append([x])
            z_list.append([x])
        else:
            a_list.append(x)
            z_list.append(x)

        out_layer = x

        for l in range(1, self.num_layers):
            activation = self.activations[l]

            input_layer = out_layer
            out_layer = np.matmul(self.weights[l - 1], input_layer)
            out_layer = np.add(out_layer, np.transpose(self.biases[l - 1])[0])
            z_list.append(out_layer)
            out_layer = activation(out_layer)
            a_list.append(out_layer)

        # NOW WE CALCULAE ERROR AT LAST LAYER AND THEN WE BACKPROPAGATE IT

        actual_layer = self.num_layers -1

        d_squared_loss = dSquaredLoss(a_list[actual_layer], y)

        d_activation = prime_function(self.activations[actual_layer])

        z_prime = d_activation(z_list[actual_layer])

        sigma = deque()

        sigma.appendleft(np.multiply(d_squared_loss, z_prime))

        bias_grad.append(np.array(sigma[0]))

        weight_grad.append(np.array(gradient_multiplier(a_list[actual_layer - 1], sigma[0])))

        actual_layer = actual_layer - 1



        while actual_layer > 0:
            d_squared_loss = np.matmul(np.transpose(self.weights[actual_layer]),sigma[0])
            d_activation = prime_function(self.activations[actual_layer])
            z_prime = d_activation(z_list[actual_layer])
            sigma.appendleft(np.multiply(d_squared_loss, z_prime))
            bias_grad.appendleft(np.array(sigma[0]))
            weight_grad.appendleft(np.array(gradient_multiplier(a_list[actual_layer - 1], sigma[0])))
            actual_layer = actual_layer - 1
        return (list(weight_grad), list(bias_grad))













    def evaluate(self, data):
        """
        :param data: dataset, a list of tuples (x, y) representing the training inputs and the desired outputs.
        :return: the number of correct predictions of the current ANN on the input dataset.
                The prediction of the ANN is taken as the argmax of its output
        """
        sum = 0
        for i in data:
            if np.argmax(self.inference(i[0])) == i[1]:
                sum = sum + 1
        return sum






data = list(load_data())
cnn = Network([784, 100, 10], [None, sigmoid, sigmoid])

training = list(data[0])

validation = list(data[1])

test = list(data[2])

i = 0

while i < len(training):

    x = np.transpose(training[i][0])[0]
    y = np.transpose(training[i][1])[0]
    training[i] = (x,y)
    i = i + 1
i = 0
while i < len(validation):
    tmp = validation[i]

    x = np.transpose(validation[i][0])[0]
    y = validation[i][1]
    validation[i] = (x,y)
    i = i + 1

#print(len(training))

#cnn.training(training, 50, 50, 3, 0, validation, 20)

#cnn.evaluate(validation)




