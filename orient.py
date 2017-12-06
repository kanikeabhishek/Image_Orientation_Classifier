import random
import numpy as np


##### Neural Network


# read text file
def read_data(filename):
    """
    :param filename: file name
    :return: list of lists including 1 class column(0,90,180,270) and 192d features, converted into integer
    """
    with open(filename, 'r') as file:

        #for row in file:
            #each_data = map(int, row.split()[2:])
            #print len(each_data)
        dataset =  [map(int, row.split()[1:]) for row in file]
        for row in dataset:
            row.append(row.pop(0))
        return dataset
        #print type(dataset),len(dataset)

# extract X and y
def extract_feature_class(dataset):
    """
    :param dataset: extract X and y from dataset read from file
    :return: list of X and list of Y
    """
    X=[]
    y=[]
    for each_data in dataset:
        X.append(each_data[1:])
        y.append(each_data[0])
    return X,y


# initialize a network
def initialize_NN(n_hiddenlayer):
    """
    :param n_hiddenlayer: number of hidden layer
    :return: list of lists, each list is a layer
    """
    return [[] for x in range(n_hiddenlayer+1)]  # hidden layer + 1 output layer


def initialize_layer_weight(input_neuron, neuron):
    """
    :param input_neuron: number of input neurons of a layer
    :param neuron: number of output neurons of a layer
    :return:
    """
    return [{'w': [random.random() for x in range(input_neuron + 1)]} for x in range(neuron)]


def construct_network(nnetwork, input_neuron, output_neuron, *hidden_neuron):
    """
    :param network: initial network with no weight
    :param input_neuron: number of neuron in input layer
    :param output_neuron: number of neuron in output layer
    :param hidden_neuron: number of neuron in hidden layer (length should equals to number of layers)
    :return: neural network with weights and bias
    """

    # if no hidden layer, only output layer
    if len(nnetwork) ==1:
        nnetwork[0] = initialize_layer_weight(input_neuron, output_neuron)
    # 1 hidden layer
    elif len(nnetwork) ==2:
        nnetwork[0] = initialize_layer_weight(input_neuron, hidden_neuron[0])
        nnetwork[1] = initialize_layer_weight(hidden_neuron[0], output_neuron)
    # multilayer
    else:
        for i in range(len(nnetwork)):
            if i ==0:
                nnetwork[i]=initialize_layer_weight(input_neuron,hidden_neuron[i])
            elif i == len(nnetwork)-1:
                nnetwork[i] = initialize_layer_weight(hidden_neuron[i], output_neuron)
            else:
                nnetwork[i] = initialize_layer_weight(hidden_neuron[i-1], hidden_neuron[i])
    return nnetwork


def activation_function(inputs, weights, para='sigmoid'):
    """
    :param inputs: input data
    :param weights: weight of neuron
    :param para: name of activation function: 'sigmoid', 'tanh', or 'Relu'.
           Default to sigmoid
    :return:
    """
    summation = weights[-1]
    for i in range(len(weights) - 1):
        summation += weights[i] * inputs[i]

    # sigmoid transfer
    if para == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-summation))

    # tanh transfer
    elif para == 'tanh':
        return np.tanh(summation)

    # ReLu transfer
    elif para == 'relu':
        return summation * (summation > 0)


def propagate_forward(nnetwork, each_data):
    """
    :param nnetwork: neural network with weights
    :param each_data: each row of data, includes X, and y at the end of the list
    :return:
    """
    inputs = each_data
    for layer in nnetwork:
        new_inputs = []
        for neuron in layer:
            # calculate the output value of a neuron
            neuron['output'] = activation_function(inputs,neuron['w'])
            new_inputs.append(neuron['output'])
        # output of a layer is the input of next layer
        inputs = new_inputs
    outputs = inputs
    return outputs
train = read_data('test-data.txt')
extract_feature_class(train)

nn = initialize_NN(1)
nn = construct_network(nn,2,2,1)

print propagate_forward(nn,[1,0,'Y'])