import tensorflow as tf
import numpy as np

class ConvPoolLayer:
    """ Definition of a combined convolutional and max-pooling layer """
    def __init__(self,input_shape,filter_shape,filter_stride,pool_shape,pool_stride=None,linear_output=False):
        """
        Parameters:
        input_shape - (mini_batch_size, rows, cols, #_feature_maps)
        filter_shape - (rows, cols, #_resulting_feature_maps)
        pool_shape - (rows, cols)
        """
        # Store arguments
        self.input_shape=input_shape
        self.filter_shape=filter_shape
        self.filter_stride=filter_stride
        self.pool_shape=pool_shape

        if pool_stride is None:
            self.pool_stride=min(self.pool_shape[0],self.pool_shape[1])
        else:
            self.pool_stride = pool_stride

        self.linear_output = linear_output

        # Define variables/parameters for the feature map
        r,c,k2=self.filter_shape
        mb,ri,ci,k1=self.input_shape
        self.shared_weights=tf.Variable(np.random.standard_normal((r,c,k1,k2)).astype(np.float32))
        # self.shared_biases = tf.Variable(np.random.standard_normal((mb,ri-r+1,ci-c+1,k2),dtype=np.float32))
        self.shared_biases = tf.constant(np.random.standard_normal((mb,ri-r+1,ci-c+1,k2)).astype(np.float32))

    def get_input_shape(self):
        """ Function to return the input Tensor dimensions """
        return self.input_shape

    def get_output_shape(self):
        """ Function to calculate and return the output Tensor dimensions """
        # Calculation borrowed from TensorFlow
        mb,r,c,k1=self.input_shape
        rf,cf,k2=self.filter_shape
        h1=int(np.ceil((r-rf+1)/self.filter_stride))
        w1=int(np.ceil((c-cf+1)/self.filter_stride))
        rp,cp=self.pool_shape
        h2=int(np.ceil((h1-rp+1)/self.pool_stride))
        w2=int(np.ceil((w1-cp+1)/self.pool_stride))
        if not self.linear_output:
            return (mb,h2,w2,k2)
        else:
            return (mb,h2*w2*k2)

    def calc_output(self,inpt):
        """
        Input is sent one mini-batch at a time.
        Dimensions of input are (mini_batch_size, rows, cols, #_feature_maps).
        An output Tensor is returned, with dimensions as specified by `self.get_output_shape()`
        """
        # Perform convolution
        _conv_output = tf.nn.conv2d(inpt,self.shared_weights,[1,self.filter_stride,self.filter_stride,1],padding='VALID')
        # Add biases
        _biased_conv_output = tf.add(_conv_output,self.shared_biases)
        # Apply activation
        _a = tf.nn.sigmoid(_biased_conv_output)
        # Perform pooling (max-pooling is default)
        _pool_output = tf.nn.max_pool(_a,[1,self.pool_shape[0],self.pool_shape[1],1],[1,self.pool_stride,self.pool_stride,1],padding='VALID')
        # Return output of correct shape
        if self.linear_output:
            return tf.reshape(_pool_output,self.get_output_shape())
        else:
            return _pool_output

class ConnectedLayer:
    """ Definition of a fully connected layer """
    def __init__(self,n_in,n_out,mini_batch_size,activation=tf.nn.sigmoid):
        # Store arguments
        self.n_in = n_in
        self.n_out = n_out
        self.mini_batch_size = mini_batch_size
        self.activation = activation

        # Create variables
        self.weights = tf.Variable(np.random.standard_normal([self.n_in,self.n_out]).astype(np.float32))
        self.biases = tf.Variable(np.random.standard_normal([1,self.n_out]).astype(np.float32))

    def get_input_shape(self):
        """ Function to return the input Tensor dimensions """
        return (self.mini_batch_size,self.n_in)

    def get_output_shape(self):
        """ Function to return the output Tensor dimensions """
        return (self.mini_batch_size,self.n_out)

    def calc_output(self,inpt):
        """
        Input is sent one mini-batch at a time.
        Dimensions of input are (mini_batch_size, #_input_neurons).
        An output Tensor is returned, with dimensions as (mini_batch_size, #_output_neurons)
        """
        # Compute weighted input and then output
        _z = tf.matmul(inpt,self.weights) + self.biases
        _output = self.activation(_z)
        return _output

class SoftmaxOutputLayer:
    """ Definition of a fully connected output layer with softmax """
    def __init__(self,n_in,n_out,mini_batch_size):
        # Store arguments
        self.n_in = n_in
        self.n_out = n_out
        self.mini_batch_size = mini_batch_size

        # Create variables
        self.weights = tf.Variable(np.random.standard_normal([self.n_in,self.n_out]).astype(np.float32))
        self.biases = tf.Variable(np.random.standard_normal([1,self.n_out]).astype(np.float32))

    def get_input_shape(self):
        """ Function to return the input Tensor dimensions """
        return (self.mini_batch_size,self.n_in)

    def get_output_shape(self):
        """ Function to return the output Tensor dimensions """
        return (self.mini_batch_size,self.n_out)

    def calc_output(self,inpt):
        """
        Input is sent one mini-batch at a time.
        Dimensions of input are (mini_batch_size, #_input_neurons).
        An output Tensor is returned, with dimensions as (mini_batch_size, #_output_neurons)
        """
        # Compute weighted input and then output
        _z = tf.matmul(inpt,self.weights) + self.biases
        _output = tf.nn.softmax(_z)
        return _output
