import const
import main
import data
import keras
from keras.ops import softmax as S
import numpy as np
from numpy import array as a
import time
import random

mdnum = "10" #main.mdnum
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

mdl.evaluate(data.file_generator("validation"))
#seed_text = "If your model has multiple outputs, you can specify different losses and metrics for each output, and you can modulate the contribution of each output to the total loss of the model."
# Source: Keras API

text = data.arr("\n"*data.rlens)
#data.arr(seed_text[-data.rlens:])
for _ in range(2000):
    append = plug(text)[-1]
    print(data.interpret(append), end="", flush=True)
    text[:-1] = text[1:]
    text[-1] = append
print()
