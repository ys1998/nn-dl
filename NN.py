"""
Implementation of a neural network class from scratch.

The network is built on sigmoid neurons, using SGD for optimization and mean square error function for cost calculation.
"""
import numpy as np
import random
import itertools


def sigmoid(z):
	return 1.0/(1.0+np.exp(-1.0*z))

def derivative_sigmoid(z):
	temp=sigmoid(z)
	return np.multiply(temp,1-temp)

def one_hot_vector(size,pos):
	temp=np.zeros(size)
	temp[pos]=1
	return np.transpose([temp])

class NN:
	def __init__(self, n_neurons):
		self.n_layers=len(n_neurons)
		self.n_neurons=n_neurons
		self.b=[np.random.randn(size,1) for size in n_neurons[1:]]
		self.w=[np.random.randn(size_next,size_curr) for size_curr,size_next in zip(n_neurons[:-1],n_neurons[1:])]

	def calc_output(self,input_values):
		if len(input_values)==self.n_neurons[0]:
			z=input_values
			for i in range(self.n_layers-1):
				z=sigmoid(np.dot(self.w[i],z)+self.b[i])
			return z

	def test(self,test_data):
		""" Function to calculate the accuracy of the NN over the provided test_data """
		if test_data:
			correct_answers=0.0
			for x,y in test_data:
				output=self.calc_output(x)
				if np.argmax(output)==y:
					correct_answers+=1
			return correct_answers/len(test_data)


	def train(self,training_data,learning_rate=1.0,mini_batch_size=1,n_epochs=10,validation_data=None):
		"""
		Function which trains the neural network with the provided training data by updating the weights and biases using stochastic gradient descent in order to minimize the mean squared error.

		If mini-batch size isn't provided, this function trains the parameters by the on-line/iterative algorithm.
		"""

		if training_data is not None:
			for epoch_no in range(n_epochs):
				""" Partition training data into mini-batches (each training data-point is an [input,true_output] pair) """
				random.shuffle(training_data)

				for batch_no in range(0, len(training_data), mini_batch_size):
					print("\33[2KEpoch : {0}, Mini-batch : {1}\r".format(epoch_no+1,int(batch_no/mini_batch_size+1)),end='')
					""" Initialize matrices to accumulate the changes in weights and biases """
					weight_errors=[np.zeros(np.shape(x)) for x in self.w]
					bias_errors=[np.zeros(np.shape(x)) for x in self.b]

					""" Iterate over all training points in the mini-batch """
					for i in range(mini_batch_size):
						j=batch_no+i
						if j>=len(training_data):
							break
						x=training_data[j][0]
						y=training_data[j][1]
						#print(x,y)
						activations=[x]
						weighted_inputs=[]
						delta=[]
						for layer_index in range(self.n_layers-1):
							weighted_inputs.append(np.dot(self.w[layer_index],x)+self.b[layer_index])
							x=sigmoid(weighted_inputs[-1])
							activations.append(x)

						""" Backpropagation algorithm """
						""" Initialize deltas for last layer """
						delta_L=np.multiply(np.subtract(activations[-1],one_hot_vector(10,y)),derivative_sigmoid(weighted_inputs[-1]))
						delta.append(delta_L)
						#print(np.shape(delta_L))

						""" Find deltas for other layers by propagating backwards """
						for l in range(self.n_layers-2,0,-1):
							delta_l=np.multiply(np.dot(np.transpose(self.w[l]),delta[0]),derivative_sigmoid(weighted_inputs[l-1]))
							delta.insert(0,delta_l)
							#print(np.shape(delta_l))

						for nw in range(len(weight_errors)):
							r,c=np.shape(weight_errors[nw])
							weight_errors[nw]=np.add(weight_errors[nw], np.dot(delta[nw][:r],np.transpose(activations[nw][:c])) )
							bias_errors[nw]=np.add(bias_errors[nw],delta[nw])

					""" Update the weights and biases """
					self.w=np.add(self.w,-learning_rate/mini_batch_size*np.array(weight_errors))
					self.b=np.add(self.b,-learning_rate/mini_batch_size*np.array(bias_errors))

				""" Testing the current parameters on the training/validation data provided """
				if not validation_data:
					print("\33[2KEpoch {0} complete.\nTraining data accuracy = {1}".format(epoch_no+1,self.test(training_data)))
				else:
					print("\33[2KEpoch {0} complete.\nTraining data accuracy = {1}\tValidation data accuracy = {2}".format(epoch_no+1,self.test(training_data),self.test(validation_data)))
