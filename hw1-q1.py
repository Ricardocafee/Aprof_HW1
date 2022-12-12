#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        #Q1.1a

        scores = np.dot(self.W, x_i.T) 
        y_hat = scores.argmax(axis=0)   
        if(y_hat != y_i):
            self.W[y_i,:]=self.W[y_i,:]+x_i
            self.W[y_hat,:]=self.W[y_hat,:]-x_i

        


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        
        scores = np.dot(self.W, x_i.T) 
        sof_max=(np.exp(scores)/np.exp(scores).sum())

        one_hot_vector = np.zeros((10))
        one_hot_vector[y_i] = 1

        diff = sof_max-one_hot_vector.T

        diff = np.reshape(diff,(10,1))
        x_i = np.reshape(x_i,(len(x_i),1))

        prod = np.dot(diff,x_i.T)

        self.W = self.W-learning_rate*prod



class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1,0.1,size=(hidden_size,n_features))
        self.W2 = np.random.normal(0.1,0.1,size=(n_classes,hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        #scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        z_i = np.dot(self.W1, X.T)+self.b1[:,None]
        h_i = self.relu(z_i)
        o_i = np.dot(self.W2, h_i)+ self.b2[:,None]
        scores=softmax(o_i)
        predicted_labels = scores.argmax(axis=0)   
        return predicted_labels


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def relu(self, z_i):
        return np.maximum(0.0, z_i)

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):

            #Forward pass
            z_i = np.dot(self.W1, x_i.T)+self.b1  
            h_i = self.relu(z_i)
            o_i = np.dot(self.W2, h_i.T) + self.b2
            soft_max = softmax(o_i)

            one_hot_vector = np.zeros((10))
            one_hot_vector[y_i] = 1

            #cost_function = -(np.multiply(one_hot_vector,np.log(soft_max)) + np.multiply((1-one_hot_vector),np.log(1-soft_max)))
            d_out = soft_max-one_hot_vector   #Partial derivative of the loss function, in order of z

            d_final = np.dot(np.reshape(d_out,(len(d_out),1)),np.reshape(h_i,(len(h_i),1)).T)  #Partial derivative in order of each weight

        
            d_relu =  np.greater(h_i, 0).astype(int)                    #Derivative relu
            sum_d = np.dot(self.W2.T,np.reshape(d_out,(len(d_out),1)))  #Sum of derivatives (hidden layer to output layer)
            d_z = np.multiply(d_relu,np.squeeze(sum_d))                 #Element-wise operation

            d_hidden = np.dot(np.reshape(x_i,(len(x_i),1)),np.reshape(d_z,(len(d_z),1)).T)

            self.W2 -= learning_rate*d_final
            self.b2 -= learning_rate*d_out

            self.W1 -= learning_rate*d_hidden.T
            self.b1 -= learning_rate*d_z



def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
