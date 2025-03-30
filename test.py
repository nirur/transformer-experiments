import main
import data
import keras
import numpy as np
from numpy import array as a
import time

mdnum = main.mdnum
mdl = keras.saving.load_model(f"saved-models/{mdnum}.keras")

#text = input("> ")
text = "\n"*data.rlens
text = data.arr(text)
for i in range(data.rlens):
    text[i, :] = mdl(a([text]))[0, i, :]
first = True
for i in range(1000):
    text[:-1] = text[1:]
    text[-1] = mdl(a([text]))[0, -1]
    if not first:
        print(data.interpret(text[-1, :]), end="", flush=True)
    else:
        print(data.toString(text), end="", flush=True)
        first = False
print()
