"""
Implementation of a deep neural network of specified structure using TensorFlow.
Current setup : Xavier Glorot initialization, ReLU activation, 'cross entropy' cost function (with softmax for last layer) and Gradient Descent optimizer
Accuracy : ~ 96 %

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
    """ Function to return groups of size `size` along with their correct outputs """
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
        self.b=[tf.Variable(tf.truncated_normal([1,x],stddev=0.1)) for x in layer_sizes[1:]]
        self.w=[tf.Variable(tf.truncated_normal([j,k],stddev=0.1)) for j,k in zip(layer_sizes[:-1],layer_sizes[1:])]

        # Define placeholders
        self.inp=tf.placeholder(tf.float32,[None,layer_sizes[0]])
        self.correct_output=tf.placeholder(tf.float32,[None,layer_sizes[-1]])

        # Define the computational graph
        self.result=tf.sigmoid(tf.add(tf.matmul(self.inp,self.w[0]),self.b[0]))
        logits=None
        for i in range(1,self.n_layers-1):
            # remove this `if` statement and keep only the `else` part when using mean square
            if i==self.n_layers-2:
                # store only the weighted input for last layer
                logits=tf.add(tf.matmul(self.result,self.w[i]),self.b[i])
                """ Sigmoid activation function """
                # self.result=tf.nn.softmax(logits)
                """ ReLU activation function """
                self.result=tf.nn.relu(logits)
            else:
                """ Sigmoid activation function """
                # self.result=tf.sigmoid(tf.add(tf.matmul(self.result,self.w[i]),self.b[i]))
                """ ReLU activation function """
                self.result=tf.nn.relu(tf.add(tf.matmul(self.result,self.w[i]),self.b[i]))

        # Compute loss/error
        """ Cross entropy loss function """
        self._cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.correct_output))
        """ Mean square loss function """
        # self._cost=tf.reduce_mean(tf.square(self.correct_output-self.result))

        # Define the training action
        self.train_step=None

        print(tf.trainable_variables())
        self._init=tf.global_variables_initializer()

    def train(self, training_data, mini_batch_size=50, n_epochs=50, learning_rate=3.0):
        """ Gradient Descent optimizer """
        self.train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(self._cost)
        """ Adam optimizer """
        # self.train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.result,axis=1),tf.argmax(self.correct_output,axis=1)),tf.float32))

        # Get training, validation and test data matrices
        i_tr,o_tr,i_va,o_va,i_te,o_te=ml.get_matrices()

        sess=tf.Session()
        sess.run(tf.initialize_all_variables())

        # initialize writer for using TensorBoard
        tf.summary.scalar("Training Accuracy", accuracy)
        tf.summary.scalar("Cost", self._cost)
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs", graph=sess.graph)

        for epoch_no in range(n_epochs):
            cntr=1
            print("Epoch number {0}".format(epoch_no+1))
            for batch,batch_output in partition_dataset(training_data,mini_batch_size):
                _,summary=sess.run([self.train_step,summary_op],{self.inp:batch, self.correct_output:batch_output})
                writer.add_summary(summary,epoch_no)
                print("\33[2K Mini-batch = {0}. Accuracy = {1}\r".format(cntr,sess.run(accuracy,{self.inp:batch, self.correct_output:batch_output})),end='',flush=True)
                cntr+=1

            # Display accuracies over training and validation data
            print("")
            print("Training data accuracy = {0}".format(sess.run(accuracy,{self.inp:i_tr, self.correct_output:o_tr})))
            print("Validation data accuracy = {0}".format(sess.run(accuracy,{self.inp:i_va, self.correct_output:o_va})))

        print("\nTest data accuracy = {0}".format(sess.run(accuracy,{self.inp:i_te, self.correct_output:o_te})))
        sess.close()
