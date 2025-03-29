import const
import datasets as ds
import random
import numpy as np
import keras

dset = ("roneneldan/TinyStories",)
#dset = ("HuggingFaceFW/fineweb",)
rlens = 20
samples = 2_000_000

mncap = 32
mxcap = 122
span = mxcap-mncap+1

def onehot(c):
    o = min(max(ord(c),mncap),mxcap)
    out = np.zeros(shape=(span,))
    out[o-mncap] = 1
    return out

def arr(seq):
    return np.array([onehot(s) for s in seq])

def data_generator(split="train"):
    dst = ds.load_dataset(*dset, split=split)
    dst = dst.shuffle()
    n = 0
    for i in dst:
        if n==samples:
            break
        i = i['text']
        #seq = np.array([onehot(j) for j in i])
        l = len(i)
        #for j in range(l-rlens):
        #    X = seq[j:j+rlens+1].copy()
        #    y = X[-1].copy()
        #    X[-1] = 0
        #    yield X, y
        if l<=rlens:
            continue
        inds = random.choices(range(l-rlens+1), k=5)
        X = np.array([
            arr(i[j:j+rlens])
            for j in inds
        ])
        y = X[:, -1, :].copy()
        X[:, -1, :] = 0
        yield X, y
        n += 1

def interpret(charr):
    ind = np.argmax(charr)
    return chr(ind+mncap)

def toString(word):
    s = ""
    for character in word:
        s += interpret(character)
    return s[-100:]
