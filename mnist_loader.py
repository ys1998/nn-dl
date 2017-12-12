import _pickle as cPickle
import numpy as np
import gzip

def load_data():
    f = gzip.open('data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='latin-1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = tr_d[1]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (list(training_data), list(validation_data), list(test_data))

# Specific functions for tf_DNN.py
def one_hot_vector(size,pos):
    ans=np.zeros([1,size])
    ans[0,pos]=1
    return ans

def get_matrices():
    tr_d, va_d, te_d = load_data()

    tr_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    tr_results = [one_hot_vector(10,i) for i in tr_d[1]]

    va_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    va_results = [one_hot_vector(10,i) for i in va_d[1]]

    te_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    te_results = [one_hot_vector(10,i) for i in te_d[1]]
    return np.transpose(np.concatenate(tr_inputs,axis=1)),np.concatenate(tr_results,axis=0),np.transpose(np.concatenate(va_inputs,axis=1)),np.concatenate(va_results,axis=0),np.transpose(np.concatenate(te_inputs,axis=1)),np.concatenate(te_results,axis=0)
