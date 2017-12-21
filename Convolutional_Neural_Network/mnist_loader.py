import _pickle as cPickle
import numpy as np
import gzip

def load_data():
    f = gzip.open('../data/MNIST/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f,encoding='latin-1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (28, 28)) for x in tr_d[0]]
    training_results = tr_d[1]
    training_data = [training_inputs, training_results]
    validation_inputs = [np.reshape(x, (28, 28)) for x in va_d[0]]
    validation_data = [validation_inputs, va_d[1]]
    test_inputs = [np.reshape(x, (28, 28)) for x in te_d[0]]
    test_data = [test_inputs, te_d[1]]
    return (training_data, validation_data, test_data)
