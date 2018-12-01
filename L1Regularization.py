from Network import *
def sign(x):
    return np.array([-1*(i <0) + (i >0) for i in x ])

def trainingL1(self, trainData, T, n, alpha, lmbda, validation, patience=10, isL1= False):
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
        print("epoch: " + str(i))
        print(max_acc)
        shuffle(trainData)
        batch_num = 0
        while (batch_num+1) * n <= len(trainData):
            updateWeightsL1(self, trainData[batch_num*n:((batch_num+1)*n)], alpha, lmbda)
            batch_num = batch_num + 1
        if batch_num*n != len(trainData):
            updateWeightsL1(self, trainData[batch_num * n:(len(trainData))], alpha, lmbda)
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



def updateWeightsL1(self, batch, alpha, lmbda=0):
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
    z = 1 - alpha * lmbda / len(batch)
    j = 0
    while j < len(self.weights):
        self.weights[j] =  self.weights[j] - alpha * res_weight[j]\
                           - alpha * lmbda * sign(self.weights[j])
        j += 1
    j = 0

    while j < len(self.biases):
        self.biases[j] = self.biases[j] - alpha * res_bias[j]\
                         + alpha*lmbda* sign(self.biases[j])
        j += 1