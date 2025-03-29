import main
import data
import keras
import numpy as np

mdnum = main.mdnum
mdl = keras.saving.load_model(f"saved-models/{mdnum}.keras")

#text = input("> ")
text = "Once upon a time, there was a "
overallS = text
text = text[-data.rlens:]
assert len(text)==data.rlens
text = np.expand_dims(data.arr(text), axis=0)
print(data.toString(text[0, :, :]))
for i in range(50):
    if not i%10: print(i)
    text = np.array(mdl(text))
    overallS += data.toString(text[:, -1, :])
print(overallS)
#mdl.evaluate(data.data_generator("validation"))
