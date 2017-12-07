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
import mnist_loader as ml

def one_hot_vector(size,pos):
    ans=np.zeros([1,size])
    ans[0,pos]=1
    return ans

def partition_dataset(training_data,size):
    """ Generator to yield groups of size `size` along with their correct outputs """
    random.shuffle(training_data)
    inp=[]; out=[]; cntr=1;
    inp.append(training_data[0][0])
    out.append(one_hot_vector(10,training_data[0][1]))
    for i in range(1,len(training_data)):
        if cntr>=size:
            cntr=0
            inp.append(training_data[i][0])
            out.append(one_hot_vector(10,training_data[i][1]))
        else:
            inp[-1]=np.concatenate([inp[-1],training_data[i][0]],axis=1)
            out[-1]=np.concatenate([out[-1],one_hot_vector(10,training_data[i][1])],axis=0)
            cntr+=1

    return list(zip([np.transpose(x) for x in inp],out))


class DNN:
    def __init__(self,layer_sizes):
        self.n_layers=len(layer_sizes)

        # Define variables
        self.b=[tf.Variable(tf.random_normal([1,x])) for x in layer_sizes[1:]]
        self.w=[tf.Variable(tf.random_normal([j,k])) for j,k in zip(layer_sizes[:-1],layer_sizes[1:])]

        # Define placeholders
        self.inp=tf.placeholder(tf.float32,[None,layer_sizes[0]])
        self.correct_output=tf.placeholder(tf.float32,[None,layer_sizes[-1]])

        # Define the computational graph
        self.result=tf.add(tf.matmul(self.inp,self.w[0]),self.b[0])
        for i in range(1,self.n_layers-1):
            self.result=tf.add(tf.matmul(self.result,self.w[i]),self.b[i])

        # Compute loss/error
        self._cost=tf.reduce_mean(-1*self.correct_output*tf.log(self.result))

        # Define the training action
        self.train_step=None

        self._init=tf.global_variables_initializer()

    def train(self, training_data, mini_batch_size=50, n_epochs=10, learning_rate=3.0):
        self.train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(self._cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.result,axis=1),tf.argmax(self.correct_output,axis=1)),tf.float32))
        sess=tf.Session()
        sess.run(self._init)
        for epoch_no in range(n_epochs):
            cntr=1
            print("Epoch number {0}".format(epoch_no+1))
            for batch,batch_output in partition_dataset(training_data,mini_batch_size):
                sess.run(self.train_step,{self.inp:batch, self.correct_output:batch_output})
                print("Mini-batch = {0}. Accuracy = {1}".format(cntr,sess.run(accuracy,{self.inp:batch, self.correct_output:batch_output})))
                cntr+=1

            # Display accuracies over training and validation data
            i,o=ml.get_training_data_matrix()
            print("Training data accuracy = {0}".format(sess.run(accuracy,{self.inp:i, self.correct_output:o})))
            # i1,o1=partition_dataset(validation_data,len(validation_data))
            # print("Validation data accuracy = {0}".format(sess.run(accuracy,{self.inp:i1, self.correct_output:o1})))
        sess.close()
