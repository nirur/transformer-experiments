import const
import datasets as ds
import numpy as np
import keras

proportion = 0.00001
dset = ("roneneldan/TinyStories",)
rlens = 30

mncap = 32
mxcap = 122

def load(split):
    dst = ds.load_dataset(*dset, split=split)
    dst = dst.to_list()
    return dst

def extractText(dst):
    txs = []
    for para in dst:
        tx = para['text']
        if len(tx)<rlens:
            continue
        txs.append(tx)
    return txs

def arr(seq):
    out = np.zeros((rlens, mxcap-mncap+1))
    for j,c in enumerate(seq):
        o = ord(c)
        if o<mncap:
            o = mncap
        elif o>mxcap:
            o = mxcap
        out[j][o-mncap] = 1
    return out

def segment(dst):
    lns = [len(i) for i in dst]
    sgs = sum(lns)-(rlens-1)*len(lns)
    out = np.zeros((sgs, rlens, mxcap-mncap+1))
    for tx in dst:
        txs = [tx[i:] for i in range(rlens)]
        for i,chs in enumerate(zip(*txs)):
            out[i] = arr(chs)
    return out

class Data(keras.utils.Sequence):
    def __init__(self, ):
        super().__init__()
    def __data_generation

def gen_dataset(split="train"):
    dst = load(split)
    dst = dst[:int(len(dst)*proportion)]
    dst = extractText(dst)
    dst = segment(dst)
    return dst

def interpret(charr):
    ind = np.argmax(charr)
    return chr(ind+mncap)

def toString(word):
    s = ""
    for character in word:
        s += interpret(character)
    return s[-100:]
