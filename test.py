import main
import data
import keras
import numpy as np

mdnum = "04" #main.mdnum
mdl = keras.saving.load_model(f"saved-models/{mdnum}.keras")

text = input("> ")
text = text[-data.rlens:]
text = " "*(data.rlens-len(text)) + text
text = data.arr(text)
print(data.toString(text))
for i in range(10):
    text[:-1] = text[1:]
    text[-1] = 0
    c = np.array(mdl(np.expand_dims(text, axis=0)))
    text[-1] = c
print(data.toString(text))
