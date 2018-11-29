def sign(x):
    return [-1*(i <0) + (i >0) for i in x ]



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
    z = 1 - alpha * lmbda / len(batch)
    j = 0
    while j < len(self.weights):
        self.weights[j] =  self.weights[j] - alpha / len(batch) * res_weight[j]\
                           + alpha*lmbda/len(batch) * sign(self.weights[j])
        j += 1
    j = 0

    while j < len(self.biases):
        self.biases[j] = self.biases[j] - alpha / len(batch) * res_bias[j]\
                         + alpha*lmbda/len(batch) * sign(self.biases[j])
        j += 1