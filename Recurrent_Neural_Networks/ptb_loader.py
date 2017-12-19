"""
Helper program to load training, test and validation data from PTB corpus.
Character and word variants available.

Entire data is available in the `data/` directory.
"""
import numpy as np

def load_words():
    path='../data/PTB/ptb.train.txt'
    words=[]
    with open(path,'r') as f:
        words.extend(f.read().replace('\n','<eos>').split())

    cntr=0; V={}
    for w in words:
        if V.get(w):
            pass
        else:
            V[w]=cntr
            cntr+=1

    return (words,V)

def load_words_raw():
    l,V = load_words()
    indices = [V[item] for item in l]
    index_to_word = {i:w for w,i in V.items()}
    return indices,index_to_word

def get_data_and_dict(data_size,batch_size,bptt_steps):
    l,V = load_words_raw()
    l=l[:data_size]
    width = batch_size
    length = len(l[:-1]) // width
    length = (length // bptt_steps)*bptt_steps
    I = np.transpose(np.reshape(l[:length*width],[width,length]))
    O = np.transpose(np.reshape(l[1:length*width+1],[width,length]))
    return I,O,V

def load_chars():
    path='data/PTB/ptb.char.train.txt'
    chars=[]
    with open(path,'r') as f:
        for line in f:
            chars.append(line.split())

    cntr=0; V={}
    for grp in chars:
        for c in grp:
            if V.get(c):
                pass
            else:
                V[c]=cntr
                cntr+=1

    return (chars,V)
