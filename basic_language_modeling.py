import ptb_loader as pl
import numpy as np
from RNN import RNN

def run():
    l,V=pl.load_words()
    # `l` is list of sentences split into words
    # `V` is a dict() mapping word with index in vocabulary
    training_data=[]
    for sent in l:
        training_data.append(pl.list_to_vector(sent,V))

    """ Initializing RNN with hidden state of dimension 100x1 """
    rnet=RNN.RNN(100,len(V))
    rnet.train(training_data)

if __name__ == '__main__':
    run()
