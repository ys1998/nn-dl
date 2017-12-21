"""
Implementation of a Convolutional Neural Network in TensorFlow.

This class provides an interface for linking the different layers which are defined in `tf_CNN_layers.py`.
Thus, CNNs of varying architecture can be created.

For details regarding these layers, have a look at the `tf_CNN_layers.py` file.
"""

import tensorflow as tf
import numpy as np
import random

def one_hot(size,pos):
    ans=np.zeros(size)
    ans[pos]=1
    return ans

class tf_CNN:
    def __init__(self, layers):
        self.layers=layers

        # Check for consistency
        for l1,l2 in zip(self.layers[:-1],self.layers[1:]):
            # print(l1.get_output_shape(),l2.get_input_shape())
            if not l1.get_output_shape()==l2.get_input_shape():
                print("Input/output dimensions are not consistent.")
                exit()

        # Define placeholders
        self.input=tf.placeholder(tf.float32,self.layers[0].get_input_shape())
        self.correct_output=tf.placeholder(tf.float32,self.layers[-1].get_output_shape())

        # Get mini-batch size
        mb_size = self.layers[0].get_input_shape()[0]

        # Define computational graph
        _intermediate_output = self.layers[0].calc_output(self.input)
        for i in range(1,len(self.layers)):
            _intermediate_output = self.layers[i].calc_output(_intermediate_output)

        # Find loss
        self._loss = tf.losses.log_loss(labels=self.correct_output,predictions=_intermediate_output)

        # Find accuracy
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(_intermediate_output,axis=1),tf.argmax(self.correct_output,axis=1)),tf.float32))

    def train(self,training_data,learning_rate=1.0, mini_batch_size=50, n_epochs=30,test_data=None,validation_data=None):
        with tf.Session() as sess:
            train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self._loss)
            sess.run(tf.global_variables_initializer())
            for epoch_no in range(n_epochs):
                """
                I - array of image matrices
                O - array of correct outputs (converted to vector form if required)
                """
                I,O=training_data
                n_batches = len(I)//mini_batch_size
                for batch_no in range(n_batches):
                    x = np.stack(I[batch_no*mini_batch_size:(batch_no+1)*mini_batch_size],axis=0)
                    # Add the `#_feature_maps` dimension
                    x = np.expand_dims(x,axis=-1)
                    y = np.stack([one_hot(10,z) for z in O[batch_no*mini_batch_size:(batch_no+1)*mini_batch_size]])
                    _,curr_loss,accuracy = sess.run([train_step,self._loss,self._accuracy],feed_dict={self.input:x,self.correct_output:y})
                    print("Epoch {0} mini-batch {1} : Loss = {2:.3}, Accuracy = {3:.3}".format(epoch_no+1,batch_no+1,curr_loss,accuracy))
