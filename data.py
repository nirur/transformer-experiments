import const
import numpy as np
#import tiktoken
import datasets as ds
import pickle

val_prop = 10 # 1/val_prop = proportion of data that goes to validation
span = 65 #1024 #50257
train_tk = False

configs = [
    {
        "path": "openwebtext",
        "streaming": True,
        "split": "train",
    },
    {
        "path": "roneneldan/TinyStories",
    },
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "CC-MAIN-2024-10",
        "split": "train",
        "streaming": True,
    },
    {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "streaming": True,
    },
]

#ec = tiktoken.get_encoding("gpt2")
#class Tokenizer:
#    def __init__(self, load_from=None):
#        self.vocab = [[x] for x in range(256)]
#        if load_from:
#            with open(load_from, 'rb') as f:
#                self.vocab = pickle.load(f)
#        self.bloc = None
#    
#    def train(self, data):
#        l = []
#        for i,t in enumerate(data):
#            if i>100:
#                break
#            t = t['text']
#            t = list(t.encode('utf-8'))
#            l.append(t)
#        print(len(self.vocab))
#        while len(self.vocab) < span:
#            d = {}
#            for t in l:
#                for a,b in zip(t, t[1:]):
#                    d[a,b] = d.get((a,b), 0)+1
#            p1, p2 = max(d.keys(), key=d.get)
#            rep = len(self.vocab)
#            self.vocab.append([p1, p2])
#            i = 0
#            while i<len(l)-1:
#                if l[i]==p1 and l[i+1]==p2:
#                    l = l[:i] + [rep] + l[i+2:]
#                i+=1
#            if len(self.vocab)%64==0: print(len(self.vocab))
#        with open(const.fp_tk, 'wb') as f:
#            pickle.dump(self.vocab, f)
#    
#    def encode(self, s):
#        if not self.bloc:
#            self.bloc = self.vocab[:256] + [0]*(span-256)
#            for i in range(256, span):
#                p,q = self.vocab[i]
#                self.bloc[i] = self.bloc[p]+self.bloc[q]
#        t = list(s.encode('utf-8'))
#        for i in range(span-1, 255, -1):
#            sub = self.vocab[i]
#            ls = len(sub)
#            j = 0
#            while j<len(t):
#                if t[j:j+ls]==sub:
#                    t = t[:j] + [i] + t[j+ls:]
#                j+=1
#        oneh = np.zeros((len(t), span))
#        for ind,val in enumerate(t):
#            oneh[ind,val] = 1
#        return oneh
#    
#    def decode(self, arr):
#        vals = [np.argmax(l) for l in arr]
#        for i in range(span-1, 255, -1):
#            j = 0
#            while j<len(vals):
#                if vals[j]==i:
#                    vals = vals[:j] + self.vocab[i] + vals[j+1:]
#                j+=1
#        return bytes(vals).decode('utf-8', errors='replace')
#        #return ec.decode([np.argmax(arr)])
#        #return ec[np.argmax(arr)]
#
#enc = Tokenizer()

ec = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
class SimpleEncoder:
    def onehot(self, arr):
        ret = np.zeros((len(arr), span))
        for i,j in enumerate(arr):
            ret[i,j] = 1
        return ret
    def encode(self, text):
        return self.onehot([ec.index(i) for i in text])
    def decode(self, text):
        return [ec[np.argmax(i)] for i in text]
enc = SimpleEncoder()

def preprocess(text):
    #t = text['text']
    t = enc.encode(text)
    for j in range(0, len(t)-const.rlens-const.batch, const.batch):
        xi = []
        yi = []
        for k in range(const.batch):
            v = t[j:j+const.rlens+1]
            x = v[:-1]
            y = v[1:]
            xi.append(x)
            yi.append(y)
        #while True:
        yield np.array(xi), np.array(yi)

def loader(*dsts):
    while True:
        for d in dsts:
            for i in d:
                yield from preprocess(i)

def fetch(cfg):
    global enc
    dst = ds.load_dataset(**cfg)
    d1 = [dst.shard(val_prop, i) for i in range(1,val_prop)]
    d2 = dst.shard(val_prop, 0)
    #with open('raw-data/shakespeare.txt') as f:
    #    dst = f.read()
    #l = len(dst)
    #ind = int(l*0.99)
    #d1 = [dst[:ind]] # It's a single training sample
    #d2 = [[dst[ind:]]] # A single shard with a single training sample
    #if train_tk:
    #    enc.train(d1[-1])
    #    print('tokenizer trained')
    #else:
    #    enc = Tokenizer(const.fp_tk)
    #assert len(enc.vocab)==span
    return loader(*d1[:-1]), loader(d2)

def repeat(data):
    while True:
        yield from preprocess(data)

def fetch_file():
    with open('raw-data/shakespeare.txt') as f:
        dst = f.read()
    l = len(dst)
    ind = int(l*0.75)
    d1 = dst[:ind] # It's a single training sample
    d2 = dst[ind:] # A single shard with a single training sample
    return repeat(d1), repeat(d2)
