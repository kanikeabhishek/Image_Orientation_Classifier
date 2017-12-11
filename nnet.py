#!/usr/bin/env python2
import random
import numpy as np
import pickle
import time
#from sklearn.preprocessing import scale # only use for normalizing data


# The neural network was implement based on the algorithm on lecture slides, also we consulted following
# resources:
# https://theclevermachine.wordpress.com/tag/backpropagation-algorithm/
# http://florianmuellerklein.github.io/nn/
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
#
# Usage:
# train_model(train_file): train the neural network using train file and save to 'nnet_model.txt'
# test_model(test_file): read 'nnet_model.txt', and predict on test data
#
# Our implementation consists of three layers: 1 input, 1 hidden and 1 output layer.
# Number of neurons in input: 192
# Number of neurons in output: 4
# The size of hidden neuron is user defined

# Three activation functions are implemented: Sigmoid, Tanh and ReLU.

# stochastic gradient descent is used due to its better performance, even with a slower running time.
# The learning rate was set to 0.01 with a decay rate of 0.001. The learning rate is decreased gradually
# based on the iteration. It uses equation as follows:
# Learning_rate = Learning_rate/(1 + decay * iteration)

# In addition, momentum term was added to avoid getting stuck in a local minima.
# The momentum is set to 0.5 by default.

# The trained model (neural network instance) is saved to 'nnet_model' using pickle module.
# Use pickle.load(open("nnet_model.txt", "rb", -1)) to load the model.

# Among 3 activation function, the sigmoid gives best results after 20 iterations with 40 hidden neurons. (68.34%)
# Please see report for detailed accuracy and running time.

# By comparing the predicted results, pic with 270
# Some misclassified image:
# 9406463030.jpg
# 3847697001.jpg
# 3940396224.jpg
# 9483846588.jpg
# 39815369.jpg

# Those images are all taken outside, and include rocks or mountains.
# Those can be difficult to classify cause there are no evidences (like sky) can indicate the direction.

# Some successfully classified image
# 3978889742.jpg
# 4082824675.jpg
# 4238737977.jpg
# 4279815500.jpg
# 4393512978.jpg
#
# Those images are also taken outside, but the either captured the sky, or they have some objects (like house)



def read_data(filename):
    """
    read the data and convert the class into a 4d binary array.
    e.g  if y = 90, insert [0, 1, 0, 0], if y = 180, insert [0, 0, 1, 0]
    :param filename: file name
    :return: list of lists including 1 class list [y1,y2,y3,y4] and 192d features, converted into integer
    format:     [[[x1,x2,x3,x4......],[y1,y2,y3,y4]],
                ...
                [[x1,x2,x3,x4......],[y1,y2,y3,y4]]]
    """
    with open(filename, 'r') as file:
        # transferred_array = []
        dataset = [map(int, row.split()[1:]) for row in file]
        new_dataset = []
        for row in dataset:
            y = row.pop(0)
            if y == 0:
                row = row + [1, 0, 0, 0]
            elif y == 90:
                row = row + [0, 1, 0, 0]
            elif y == 180:
                row = row + [0, 0, 1, 0]
            elif y == 270:
                row = row + [0, 0, 1, 1]
            new_dataset.append(row)
        new_array = np.array(new_dataset)
        y = new_array[:, -4:]
        #X = scale(new_array[:, :-4])
        X = new_array[:, :-4]
        return [[X[i, :].tolist(), y[i].tolist()] for i in range(X.shape[0])]


def extract_y(filename):
    """
    :param filename: filename
    :return: direction of pic divided by 90
    eg.if y = 90, return 1
    """
    with open(filename, 'r') as file:
        return [int(row.split()[1]) / 90 for row in file]


def extract_test_pic_name(filename):
    with open(filename, 'r') as file:
        return [row.split()[0] for row in file]


class neural_network(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, activation):
        """
        :param n_inputs: # of input features
        :param n_hidden: # neurons in hidden layer
        :param n_outputs: # of classes
        :param activation: activation function. Currently supports sigmoid, tanh, ReLu
        """

        self.inputs = n_inputs +1  # plus one bias term
        self.hidden = n_hidden
        self.outputs = n_outputs

        # iteration default to 50
        self.iter = 50
        self.l_rate = 0.01    # learning rate
        self.act = activation  # activation function
        self.momentum = 0.5
        self.learning_decay = 0.001

        # initial activation layer
        self.ai = np.zeros(self.inputs)
        self.ah = np.zeros(self.hidden)
        self.ao = np.zeros(self.outputs)

        # random weights
        self.wi = np.random.normal(size=(self.inputs, self.hidden))
        self.wo = np.random.normal(size=(self.hidden, self.outputs))

    def activation_function(self,weights):
        """
        :param weights: weight of neuron
        :param para: name of activation function: 'sigmoid', 'tanh', or 'Relu'.
        :return: activated weights
        """

        # sigmoid transfer
        if self.act == 'sigmoid':
            return 1 / (1 + np.exp(-weights))

        # tanh transfer
        elif self.act == 'tanh':
            return np.tanh(weights)

        # ReLu transfer
        elif self.act == 'relu':
            return weights * (weights > 0)

    def derivative(self, transfered):
        """
        :param transfered: activated transferred data
        :return:
        """
        if self.act == 'sigmoid':
            return transfered * (1.0 - transfered)
        elif self.act == 'tanh':
            return 1.0 - transfered ** 2
        elif self.act == 'relu':
            return 1. * (transfered > 0)
            #return cp_transfered

    def propagate_forward(self, inputs):
        """
        Propagate forward
        :param inputs: input data
        :return:
        """

        # activation for input layer
        self.ai[:self.inputs - 1] = inputs
        # activation for hidden layer
        self.ah = self.activation_function(np.dot(self.wi.T, self.ai))
        # activation for output layer
        self.ao = self.activation_function(np.dot(self.wo.T, self.ah))

        return self.ao

    def propagate_backward(self, targets):
        """
        Propagate_backward
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """

        # gradient decent
        output_deltas = self.derivative(self.ao) * -(targets - self.ao)

        hidden_deltas = self.derivative(self.ah) * np.dot(self.wo, output_deltas)

        # update the weights backwards for each layer
        # output
        temp = output_deltas * np.reshape(self.ah, (self.ah.shape[0], 1))
        self.wo -= self.l_rate * (temp)+ temp*self.momentum

        # input
        temp = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        self.wi -= self.l_rate * (temp)+ temp*self.momentum

        # sum error
        error = sum((0.5*(targets - self.ao)) ** 2)

        return error

    def train(self, data):
        """
        train neural network using data
        :param data: training data
        :return:
        """

        for i in range(self.iter):
            error = 0.0
            # Shuffling training examples
            random.shuffle(data)
            for p in data:
                inputs = p[0]
                label = p[1]
                self.propagate_forward(inputs)
                error += self.propagate_backward(label)

            # learning rate decay
            self.l_rate *= 1 / (1+self.learning_decay*i)

    def predict(self, data):
        """
        :param data: test data
        :return: list of predict labels
        """
        predictions = []
        predict_label = []
        for p in data:
            predictions.append(self.propagate_forward(p[0]))
        for pre in predictions:
            predict_label.append(np.argmax(pre))

        return predict_label


def train_model(train_file,model_file):
    """
    train the neural network using train file and save to 'nnet_model.txt'
    :param train_file: train file
    :return:
    """

    # read text file
    X = read_data(train_file)

    nnet = neural_network(n_inputs=192, n_hidden=40, n_outputs=4, activation='sigmoid')
    nnet.iter = 20
    nnet.train(X)
    #neural_network.__module__ = "nnet"
    with open(model_file,'wb') as file:
        pickle.dump(nnet, file, -1)


def test_model(test_file,model_file):
    """
    read 'nnet_model.txt', and predict on test data
    :param test_file: test file
    :return:
    """
    #print model_file
    nnet = pickle.load(open(model_file, "r", -1))


    #print 'test'
    x = read_data(test_file)
    true_label = extract_y(test_file)

    predict_label = nnet.predict(x)
    pic_name = extract_test_pic_name(test_file)
    output_list = []
    for i in range(len(x)):
        output_list.append(pic_name[i]+' ' + str(predict_label[i]*90))

    with open('output.txt','wb') as outfile:
        for item in output_list:
            outfile.write("%s\n" % item)



    # calculate accuracy
    count = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    for i in range(len(x)):
        if predict_label[i] == true_label[i]:
            if true_label[i] == 0:
                count0 +=1
            elif true_label[i] == 1:
                count1 +=1
            elif true_label[i] == 2:
                count2 +=1
            elif true_label[i] == 3:
                count3 +=1
            count +=1
    #print count0,count1,count2,count3
    print 'Accuracy:', float(count)/len(x)*100
#
# trainfile = 'train-data.txt'
# testfile = 'test-data.txt'
# test_model(testfile,'nnet_model.txt')

# if __name__ == '__main__':
#     #start = time.time()
#     trainfile = 'train-data.txt'
#     testfile = 'test-data.txt'
#     #train_model(trainfile)
#     test_model(testfile,'nnet_model.txt')
#     #end = time.time()
#     #print end - start