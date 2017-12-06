"""
Implementation of a deep neural network of specified structure using TensorFlow.

It uses matrix-based batch training, with the following possible variations :
 - Gradient Descent / Adam optimizer
 - Mean square / cross-entropy cost functions
 - Sigmoid (which is similar to tanh) / ReLU / softmax activation functions

After training is complete, these variations are compared using the corresponding learning curves.
(the MNIST database shall be used for comparision)
"""

import tensorflow as tf
import numpy as np
import random

def partition_dataset(training_data,size):
    """ Generator to yield groups of size `size` along with their correct outputs """
    random.shuffle(training_data)
    
class DNN:
    def __init__(self,layer_sizes):
        self.n_layers=len(layer_sizes)
        self.b=[tf.Variable(tf.zeros(x,1)) for x in layer_sizes[1:]]
        self.w=[tf.Variable(tf.zeros(j,k)) for j,k in zip(layer_sizes[:-1],layer_sizes[1:])]
        self.input=tf.placeholder(tf.float32,[None,layer_sizes[0]])
        self.correct_output=tf.placeholder(tf.float32,[None,layer_sizes[-1]])
        self._init=tf.global_variable_initializor()

    def train(
            self,training_data,
            mini_batch_size=50,
            n_epochs=10,
            learning_rate=3.0,
            validation_data=None
        ):
        for batch,output in partition_dataset(training_data,mini_batch_size):
