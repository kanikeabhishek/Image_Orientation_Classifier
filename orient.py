#!/usr/bin/env python2

import ada_boost
from knn import KNN
from nnet import train_model, test_model
import sys

def verifyInput():
    if len(sys.argv) < 5:
        print "Usage: ./orient.py train train_file.txt model_file.txt [model]"
        print "or"
        print "Usage: ./orient.py test test_file.txt model_file.txt [model]"
        print "[model] from nearest, adaboost, nnet, best"
        sys.exit(0)

def adaBoost(train_test, input_file, model_file):
    if train_test == "train":
        ada_boost.train(input_file, model_file)
    elif train_test == "test":
        ada_boost.predict(input_file, model_file)

def nearest(train_test, input_file, model_file):
    knn_instance = KNN()
    if train_test == "train":
        knn_instance.train(input_file, model_file)
    elif train_test == "test":
        knn_instance.predict(input_file, model_file)

def n_net(train_test, input_file, model_file):
    if train_test == 'train':
        train_model(input_file,model_file)
    elif train_test == 'test':
        test_model(input_file,model_file)

def best(train_test,input_file,model_file):
    if train_test == "train":
        ada_boost.train(input_file, model_file)
    elif train_test == "test":
        ada_boost.predict(input_file, model_file)

    

def main():
    verifyInput()
    train_test, input_file, model_file, model = sys.argv[1:]
    #print train_test,input_file,model_file,model
    if model == "adaboost":
        adaBoost(train_test, input_file, model_file)
    elif model == "nearest":
        nearest(train_test, input_file, model_file)
    elif model == 'nnet':
        n_net(train_test, input_file, model_file)
    elif model =='best':
        adaBoost(train_test, input_file, model_file)



if __name__ == '__main__':
    main()
