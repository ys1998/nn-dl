import ptb_loader as pl
import numpy as np
from RNN import RNN

def run():
    l,V=pl.load_words()
    # `l` is list of sentences split into words
    # `V` is a dict() mapping word with index in vocabulary

    # Convert words to respective indices
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j]=V[l[i][j]]

    # Generate training data
    training_data=[]
    for sent in l:
        training_data.append( (sent[:-1],sent[1:]) )

    """ Initializing RNN with hidden state of dimension 20x1 """
    rnet=RNN(20,len(V))
    rnet.train(
                   training_data[:25],
                   learning_rate=3.0,
                   bptt_step=10,
                   transform=lambda sent: [pl.one_hot(len(V),x) for x in sent]
               )

if __name__ == '__main__':
    run()
