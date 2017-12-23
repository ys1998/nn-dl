"""
Implementation of Long Short-Term Memory network in TensorFlow.
"""
import tensorflow as tf
import numpy as np

def one_hot(size,pos):
    ans=np.zeros([size,1])
    ans[pos]=1
    return ans

def softmax(z):
    # Naive implementation - doesn't handle overflows
    return np.exp(z)/np.sum(np.exp(z))

def decay(min_learning_rate,max_learning_rate,frac):
    return max_learning_rate - (max_learning_rate-min_learning_rate)*frac

class tf_LSTM:
    def __init__(
                    self,
                    input_size,
                    batch_size,
                    bptt_steps,
                ):
        # Store arguments
        self._batch_size=int(batch_size)
        self._input_size=int(input_size)
        self._bptt_steps=int(bptt_steps)

        # Define variables

        # Forget Gate
        self.Wf=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Rf=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Bf=tf.Variable(tf.zeros([input_size,1]))
        # Input Gate
        self.Wi=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Ri=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Bi=tf.Variable(tf.zeros([input_size,1]))
        # Output Gate
        self.Wo=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Ro=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.Bo=tf.Variable(tf.zeros([input_size,1]))
        # State change
        self.W=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.R=tf.Variable(tf.random_uniform([input_size,input_size],-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size)))
        self.B=tf.Variable(tf.zeros([input_size,1]))

        # Placeholders
        self.input=tf.placeholder(tf.int32,[self._bptt_steps,batch_size],name="Input")
        self.correct_output=tf.placeholder(tf.int32,[self._bptt_steps,batch_size],name="Output")
        self._init_cell_state=tf.placeholder(tf.float32,[input_size,batch_size],name="Initial_Cell_State")
        self._init_hidden_state=tf.placeholder(tf.float32,[input_size,batch_size],name="Initial_Hidden_State")

        # Computations
        inp = tf.transpose(tf.one_hot(self.input[0],depth=self._input_size))
        f = tf.nn.sigmoid(tf.matmul(self.Wf,inp)+tf.matmul(self.Rf,self._init_hidden_state)+self.Bf)
        i = tf.nn.sigmoid(tf.matmul(self.Wi,inp)+tf.matmul(self.Ri,self._init_hidden_state)+self.Bi)
        o = tf.nn.sigmoid(tf.matmul(self.Wo,inp)+tf.matmul(self.Ro,self._init_hidden_state)+self.Bo)
        state_change = tf.nn.tanh(tf.matmul(self.W,inp)+tf.matmul(self.R,self._init_hidden_state)+self.B)
        self._cell_state = f*self._init_cell_state+i*state_change
        self._hidden_state = tf.nn.tanh(self._cell_state)*o
        self._loss = -tf.transpose(tf.one_hot(self.correct_output[0],depth=self._input_size))*tf.log(0.5*self._hidden_state+0.5)

        for cntr in range(1,self._bptt_steps):
            inp = tf.transpose(tf.one_hot(self.input[cntr],depth=self._input_size))
            f = tf.nn.sigmoid(tf.matmul(self.Wf,inp)+tf.matmul(self.Rf,self._hidden_state)+self.Bf)
            i = tf.nn.sigmoid(tf.matmul(self.Wi,inp)+tf.matmul(self.Ri,self._hidden_state)+self.Bi)
            o = tf.nn.sigmoid(tf.matmul(self.Wo,inp)+tf.matmul(self.Ro,self._hidden_state)+self.Bo)
            state_change = tf.nn.tanh(tf.matmul(self.W,inp)+tf.matmul(self.R,self._hidden_state)+self.B)
            self._cell_state = f*self._cell_state+i*state_change
            self._hidden_state = tf.nn.tanh(self._cell_state)*o
            self._loss += -tf.transpose(tf.one_hot(self.correct_output[cntr],depth=self._input_size))*tf.log(0.5*self._hidden_state+0.5)

        self._loss=tf.reduce_mean(self._loss)
        self._init=tf.global_variables_initializer()

    def train(self,input_data,output_data,learning_rate=1.0,n_epochs=30,factor=10):
        """
        Training data is contained in `input_data`, `output_data`.
        Both of these arrays have `batch_size` number of columns and arbitrary number of rows.
        For language modeling :
        Each element of these arrays is the index of a particular word from the vocabulary.

        `bptt_steps` is the number of steps upto which truncated BPTT will be applied.
        """
        I=input_data; O=output_data
        with tf.Session() as sess:
            sess.run(self._init)

            cell_state = np.zeros([self._input_size,self._batch_size])
            hidden_state = np.zeros([self._input_size,self._batch_size])

            for epoch_no in range(n_epochs):
                total_loss = 0.0
                cur_learning_rate = decay(learning_rate/factor,learning_rate,epoch_no/n_epochs)
                train = tf.train.GradientDescentOptimizer(learning_rate=cur_learning_rate).minimize(self._loss)
                print("Current learning rate = {0}".format(cur_learning_rate))
                for cntr in range(len(I)//self._bptt_steps):
                    _, cell_state, hidden_state, curr_loss = sess.run([train,self._cell_state,self._hidden_state,self._loss],
                            feed_dict={
                                        self.input:I[cntr*self._bptt_steps:min(len(I),(cntr+1)*self._bptt_steps),:],
                                        self.correct_output:O[cntr*self._bptt_steps:min(len(I),(cntr+1)*self._bptt_steps),:],
                                        self._init_cell_state:cell_state,
                                        self._init_hidden_state:hidden_state
                                    })
                    total_loss += curr_loss
                    print("Loss after epoch {0}, batch {1} = {2}".format(epoch_no+1,(cntr+1)*self._bptt_steps,curr_loss/self._bptt_steps))
                print("Average loss in epoch {0} = {1}".format(epoch_no+1,total_loss/len(I)))
