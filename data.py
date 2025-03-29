import const
import datasets as ds
import random
import numpy as np
import keras

dset = {
    "path": "roneneldan/TinyStories",
}
#dset = {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "streaming": True} #"CC-MAIN-2024-10"
rlens = 30

mncap = 0
mxcap = 126
span = mxcap-mncap+1

def onehot(c):
    o = min(max(ord(c),mncap),mxcap)
    out = np.zeros(shape=(span,))
    out[o-mncap] = 1
    return out

def arr(seq):
    return np.array([onehot(s) for s in seq])

def data_generator(split="train"):
    dst = ds.load_dataset(**dset, split=split)
    dst = dst.shuffle()
    for i in dst:
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
        inds = random.choices(range(l-rlens), k=10)
        v = np.array([
            arr(i[j:j+rlens+1])
            for j in inds
        ])
        x, y = v[:, :-1, :], v[:, 1:, :]
        yield x, y

def interpret(charr):
    ind = np.argmax(charr)
    return chr(ind+mncap)

def toString(word):
    s = ""
    for character in word:
        s += interpret(character)
    return s[-100:]
