from Network import *
import matplotlib.pyplot as plt
from time import time

def preprocess():
    data = list(load_data())

    training = list(data[0])

    validation = list(data[1])

    test = list(data[2])

    i = 0

    while i < len(training):
        x = np.transpose(training[i][0])[0]
        y = np.transpose(training[i][1])[0]
        training[i] = (x, y)
        i = i + 1
    i = 0
    while i < len(validation):
        x = np.transpose(validation[i][0])[0]
        y = validation[i][1]
        validation[i] = (x, y)
        i = i + 1
    i = 0
    while i < len(test):
        x = np.transpose(test[i][0])[0]
        y = test[i][1]
        test[i] = (x, y)
        i = i + 1

    return training, validation,test
def main():
    training, validation, test = preprocess()
    #showResults(training, validation, test)
    print("here")
    cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
    cnn.training(training, 30, 10, 0.1, 5/len(training),validation,20,True)








def showResults(training, validation, test):
    batch = [0.5,1,2,3,5,8,16,32,64]
    res = []
    print("Alpha varies, 10 epochs, lambda = 0")
    for i in batch:
        print("alpha = " + str(i))
        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 10, 10, i, 0, validation)
        res.append(cnn.evaluate(test))
    plt.plot(batch, res)
    plt.show()
    batch = [0.5, 1, 2, 3, 5, 8, 16, 32, 64]
    res = []
    print("Alpha varies, 30 epochs, lambda = 0")
    for i in batch:
        print("alpha = " + str(i))
        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 30, 10, i, 0, validation)
        res.append(cnn.evaluate(test))
    plt.plot(batch, res)
    plt.show()


    batch = [1, 3, 5, 8, 16]
    res = []
    print("Alpha varies, we wait for early stopping or 400 epochs, lambda = 1/len(training)")
    for i in batch:
        print("alpha = " + str(i))
        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 400, 10, i, 1/len(training), validation, 20)
        res.append(cnn.evaluate(test))
    plt.plot(batch, res)
    plt.show()

    batch = [10, 20, 30, 40, 60, 70, 80, 90, 100]
    res = []
    print("Alpha= 3, we wait for early stopping or 400 epochs, lambda = 1/len(training), hidden nodes varies")
    for i in batch:
        print("alpha = " + str(i))
        cnn = Network([784, i, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 400, 10, 3, 1 / len(training), validation, 20)
        res.append(cnn.evaluate(test))
    plt.plot(batch, res)
    plt.show()

    batch = [leaky_relu, tanh]
    res = []
    print("Alpha= 0.01, we wait for early stopping or 400 epochs, lambda = 1/len(training), activation function varies")
    for i in batch:
        print("alpha = " + str(i))
        cnn = Network([784, i, 10], [None, i, i])
        cnn.training(training, 400, 10, 0.01, 5/ len(training), validation, 20)
        res.append(cnn.evaluate(test))
    plt.plot(res)
    plt.show()

    print("Now we show the best result")
    batch = [1]
    for i in batch:
        print("activation = " + str(i))
        cnn = Network([784, 60, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 800, 10, 0.1, 5/len(training), validation, 20)
        res.append(cnn.evaluate(test))









if __name__ == '__main__':
   main()

