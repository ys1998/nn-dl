"""
Implementation of a vanilla RNN.
Uses tanh activation function, SGD for optimization, cross-entropy cost function
with softmax output layer and custom dimensions for hidden state and input.

Relations between parameters :
h[t] = tanh( Ux[t] + Wh[t-1] + B1 )
o[t] = softmax( Vh[t] + B2 )

Here, the biases B1 and B2 are optional, and are generally ignored.
"""

import numpy as np

def tanh(z):
    f1=np.exp(z); f2=np.exp(-1*z)
    return np.divide(f1-f2,f1+f2)

# def tanh_prime(z):
#     temp = tanh(z)
#     return 1.0-temp**2

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

class RNN:
    def __init__(self,state_size,input_size,ignore_bias=True):
        # List to store hidden state, initialized with a zero matrix
        self.h=[np.zeros([state_size,1])]
        # List to store output at each time step
        self.o=[]
        # List to store input at each time step
        self.x=[]
        # Parameters to be trained; standard initialization
        self.U=np.random.uniform(-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size),[state_size,input_size])
        self.V=np.random.uniform(-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size),[input_size,state_size])
        self.W=np.random.uniform(-1.0/np.sqrt(input_size),1.0/np.sqrt(input_size),[state_size,state_size])
        # Optional biases
        self.ignore_bias=ignore_bias
        if not ignore_bias:
            self.B1=np.zeros([state_size,1])
            self.B2=np.zeros([input_size,1])

        # Loss
        self._loss=0

    def _feed(self,X,Y):
        """
        Function to evaluate results for each time step.
        X - input sequence
        Y - correct output sequence
        """
        for x,y in zip(X,Y):
            self.x.append(x)
            if self.ignore_bias:
                self.h.append(tanh(np.dot(self.U,x)+np.dot(self.W,self.h[-1])))
                self.o.append(softmax(np.dot(self.V,self.h[-1])))
                self._loss+=-np.sum(y*np.log(self.o[-1]))
            else:
                self.h.append(tanh(np.dot(self.U,x)+np.dot(self.W,self.h[-1])+self.B1))
                self.o.append(softmax(np.dot(self.V,self.h[-1])+self.B2))
                self._loss+=-np.sum(y*np.log(self.o[-1]))

    def _reset(self):
        """
        Function that clears the input and output queues and reinitializes
        the hidden state keeping the other parameters unchanged
        """
        self.x=[]
        self.o=[]
        self._loss=0
        # self.h=[self.h[-1]]
        self.h=[self.h[0]]

    def _bptt(self,Y,step=-1):
        """
        Function which applies truncated backpropagation through time using the
        existing lists of input and output data, and hidden state
        Y - corresponding correct output
        """
        # Total length of sequence / total time
        T=len(self.x)
        dEdU=np.zeros(np.shape(self.U))
        dEdV=np.zeros(np.shape(self.V))
        dEdW=np.zeros(np.shape(self.W))
        if self.ignore_bias:
            for t in range(T-1,-1,-1):
                delta_o = np.subtract(self.o[t],Y[t])
                dEdV += np.dot(delta_o,np.transpose(self.h[t]))
                delta_t = np.multiply(np.dot(np.transpose(self.V),delta_o),1.0-self.h[t]**2)
                if step==-1:
                    for bps in range(t-1,-1,-1):
                        dEdW += np.outer(delta_t,self.h[bps-1])
                        dEdU += np.outer(delta_t,self.x[bps-1])
                        delta_t=np.dot(np.transpose(self.W),delta_t)*(1.0-self.h[bps-1]**2)
                else:
                    for bps in range(t-1,max(-1,t-step-1),-1):
                        dEdW += np.outer(delta_t,self.h[bps-1])
                        dEdU += np.outer(delta_t,self.x[bps-1])
                        delta_t=np.dot(np.transpose(self.W),delta_t)*(1.0-self.h[bps-1]**2)

            return [dEdU, dEdV, dEdW]
        else:
            # Initialized but used only if required
            dEdB1=np.zeros(np.shape(self.B1))
            dEdB2=np.zeros(np.shape(self.B2))
            for t in range(T-1,-1,-1):
                delta_o = np.subtract(self.o[t],Y[t])
                dEdV += np.dot(delta_o,np.transpose(self.h[t]))
                delta_t = np.multiply(np.dot(np.transpose(self.V),delta_o),1.0-self.h[t]**2)
                if step==-1:
                    for bps in range(t-1,-1,-1):
                        dEdW += np.outer(delta_t,self.h[bps-1])
                        dEdU += np.outer(delta_t,self.x[bps-1])
                        delta_t=np.dot(np.transpose(self.W),delta_t)*(1.0-self.h[bps-1]**2)
                else:
                    for bps in range(t-1,max(-1,t-step-1),-1):
                        dEdW += np.outer(delta_t,self.h[bps-1])
                        dEdU += np.outer(delta_t,self.x[bps-1])
                        delta_t=np.dot(np.transpose(self.W),delta_t)*(1.0-self.h[bps-1]**2)

            return [dEdU, dEdV, dEdW, dEdB1, dEdB2]

    def train(self,training_data,learning_rate=0.5,n_epochs=50,bptt_step=-1,transform=lambda x: x):
        """
        Function to train the RNN. All hyper-parameters are trivial, except perhaps `transform`.
        It is a container for any function to be applied to `training_data` before it is used.
        """
        for epoch_no in range(n_epochs):
            cntr=0
            # Here X and Y are sequences of words
            for org_X,org_Y in training_data:
                X=transform(org_X); Y=transform(org_Y)
                # print(X,Y)
                cntr+=1
                self._feed(X,Y)
                print("Loss in epoch {0} : batch {1} = {2}".format(epoch_no+1,cntr,self._loss))
                dEdU, dEdV, dEdW = self._bptt(Y,bptt_step)
                self.W+=-learning_rate*dEdW
                self.U+=-learning_rate*dEdU
                self.V+=-learning_rate*dEdV
                self._reset()
