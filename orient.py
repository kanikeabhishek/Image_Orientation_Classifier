#!/usr/bin/env python2

import ada_boost
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

def main():
    verifyInput()
    train_test, input_file, model_file, model = sys.argv[1:]
    if model == "adaboost":
        adaBoost(train_test, input_file, model_file)

if __name__ == '__main__':
    main()