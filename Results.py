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
    fastTrain(training, validation, test)
    showResults(training, validation, test)
    cnn = Network([784,30,10], [None, sigmoid, sigmoid])
    function_pool = [ leaky_relu]
    result = []
    for func in function_pool:
        cnn = Network([784, 30, 20, 10], [None, func, func])
        print(str(func))
        cnn.training(training, 100, 10, 0.5, 0.5, validation)
        result.append(cnn.evaluate(validation))
    plt.plot(result)
    plt.show()

def fastTrain(training, validation, test):
    alphas = [0.5, 1, 3, 5, 8, 16, 32, 64]
    res = []
    print("Now the number of epochs is 10, we check for different learning rates")
    for i in alphas:

        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        start = time()
        cnn.training(training, 1, 10, i, 0, validation, 10)
        end = time()
        print("The learning rate is: " + str(i) + " the accuracy of the test is: " + str(acc))
        print("seconds taken")
        print(end - start)
        acc = cnn.evaluate(test)
        res.append(acc)
    plt.plot(alphas,res)
    plt.show()
    res = []
    print("now we set the epochs to 30")
    for i in alphas:
        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        start = time()
        cnn.training(training, 30, 10, i, 0, validation, 10)
        end = time()
        print("The learning rate is: " + str(i) + " the accuracy of the test is: " + str(acc))
        print("seconds taken")
        print(end - start)
        acc = cnn.evaluate(test)
        res.append(acc)

    plt.plot(alphas, res)
    plt.show()





def showResults(training, validation, test):
    batch = [25, 50, 100, 200, 400, 800]
    res = []
    for i in batch:
        cnn = Network([784, 30, 10], [None, sigmoid, sigmoid])
        cnn.training(training, 500, i, 0.1, 5/len(training), validation, 500)
        res.append(cnn.evaluate(validation))
    plt.plot(res)
    plt.show()




if __name__ == '__main__':
    main()