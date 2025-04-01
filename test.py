import const
import main
import data
import keras
from keras.ops import softmax as S
import numpy as np
from numpy import array as a
import time
import random

mdnum = main.mdnum
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

#mdl.evaluate(data.file_generator("validation"))
text = np.zeros((data.rlens, data.span))
text[0, 0] = 1
for i in range(1,data.rlens):
    out = plug(text, True)
    text[i] = out[i]

print(data.interpret(text), end="", flush=True)
for i in range(2000):
    append = plug(text)[-1]
    print(data.interpret(append), end="", flush=True)
    text[:-1] = text[1:]
    text[-1] = append
print()
