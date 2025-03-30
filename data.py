import const
import datasets as ds
import random
import numpy as np
import keras

#dset = { "path": "roneneldan/TinyStories", }
#dset = {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "streaming": True} #"CC-MAIN-2024-10"
rlens = 32
batch = 16

temp = 1

#mncap = 0
#mxcap = 126
#mncap = 32
#mxcap = 122
#span = mxcap-mncap+1

s = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
enc = {c:i for i,c in enumerate(s)}
span = len(s)

def toNum(c):
    #ret = min(max(ord(c),mncap),mxcap)-mncap
    ret = enc[c]
    return ret

def fromNum(i):
    #ret = chr(ind+mncap)
    ret = s[i]
    return ret

def onehot(c):
    o = toNum(c)
    out = np.zeros(shape=(span,))
    out[o] = 1
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
        inds = random.choices(range(l-rlens), k=batch)
        v = np.array([
            arr(i[j:j+rlens+1])
            for j in inds
        ])
        x, y = v[:, :-1, :], v[:, 1:, :]
        yield x, y

def file_generator(split="train",fl="shakespeare.txt"):
    with open(fl) as f:
        dst = f.read()
    l = len(dst)
    if split=="train":
        dst = dst[:l*3//4]
    else:
        dst = dst[l*3//4:]
    dst = arr(dst)
    l = len(dst) - rlens
    inds = list(range(l))
    l2 = len(inds)
    i = 0
    while True:
        if i>=l2:
            random.shuffle(inds)
            i = 0
        v = np.array([
            dst[j:j+rlens+1]
            for j in inds[i:i+batch]
        ])
        x, y = v[:, :-1, :], v[:, 1:, :]
        yield x, y
        i += batch

def interpret(charr):
    ind = random.choices(range(span), keras.ops.softmax(charr/temp))
    return fromNum(*ind)

def toString(word):
    s = ""
    for character in word:
        s += interpret(character)
    return s[-100:]
