import const
import random
import numpy as np
import keras
import datasets as ds

rlens = 256
batch = 64

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
    #ret = chr(i+mncap)
    ret = s[i]
    return ret

def onehot(c):
    out = np.zeros(shape=(span,))
    out[toNum(c)] = 1
    return out

def arr(seq):
    return np.array([onehot(s) for s in seq])

def data_generator(dset_conf):
    dst = ds.load_dataset(**dset_conf)
    dst = dst.shuffle()
    for i in dst:
        i = i['text']
        l = len(i)
        if l<=rlens:
            continue
        for k in range(0, l-rlens-batch+1, batch):
            v = np.array([
                arr(i[j:j+rlens+1])
                for j in range(k, k+batch)
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
    while split=="train" or i < l2:
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

def _interp(charr):
    ind = np.argmax(charr)
    return fromNum(ind)

def interpret(word):
    if word.ndim==1:
        return _interp(word)
    s = ""
    for character in word:
        s += _interp(character)
    return s
