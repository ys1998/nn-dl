"""
Helper program to load training, test and validation data from PTB corpus.
Character and word variants available.

Entire data is available in the `data/` directory.
"""
import numpy as np

def one_hot(size,pos):
    ans=np.zeros([size,1])
    ans[pos]=1
    return ans

def list_to_vector(l,V):
    size=len(V)
    res=[one_hot(size,V[x]) for x in l]
    return res[:-1],res[1:]

def load_words():
    path='data/ptb.train.txt'
    words=[]
    with open(path,'r') as f:
        for line in f:
            words.append(line.split())

    cntr=0; V={}
    for sent in words:
        for w in sent:
            if V.get(w):
                pass
            else:
                V[w]=cntr
                cntr+=1

    return (words,V)

def load_chars():
    path='data/ptb.char.train.txt'
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
