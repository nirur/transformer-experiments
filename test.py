import const
import data
import model
import main
import keras
from keras.ops import softmax as S
import numpy as np
from numpy import array as a
import time
import random

mdnum = const.mdnum
mdl = keras.saving.load_model(f"saved-models/{mdnum}.keras")

def snap(arr):
    ind = random.choices(range(data.span), arr)
    arr[:] = 0
    arr[ind] = 1

def plug(curr: np.ndarray, snap_prev=False):
    fluffed_curr = a([curr])
    out = mdl(fluffed_curr)[0]
    out = S(out/const.temp)
    out = a(out)
    snap(out[-1])
    if snap_prev:
        for i in range(data.rlens-1):
            snap(out[i])
    return out

#mdl.evaluate(main.val_gen)
data.enc = data.Tokenizer(const.fp_tk)

text = np.zeros((const.rlens, data.span))
text[:, 10] = 1
for i in range(500):
    if not i%20: print(i)
    append = plug(text[-const.rlens:])[-1:]
    text = np.concatenate((text, append), axis=0)
print(data.enc.decode(text[const.rlens:]), end="", flush=True)
print()
