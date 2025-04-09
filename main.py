import const
import data
import model
import math
import numpy as np
import keras
from keras import optimizers, losses, metrics

#import jax
#from jax.experimental import mesh_utils
#from jax.sharding import Mesh
#from jax.sharding import NamedSharding
#from jax.sharding import PartitionSpec as P

def main():
    mdl = model.gen_model()
    compile_model(mdl)
    mdl.summary()
    
    train, val = data.fetch_file()#data.configs[0])
    mdl.fit(
        train,
        epochs=200,
        steps_per_epoch=500,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                const.fp,
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch",
            ),
        ],
        validation_data=val,
        validation_steps=10,
    )

#cc_k = losses.CategoricalCrossentropy(
#    from_logits=True,
#    reduction=None,
#)
#weight = np.array([n**-2 for n in range(const.rlens, 0, -1)])
#weight *= 6*math.pi**-2
#@keras.saving.register_keras_serializable()
#def cce(y_true, y_pred):
#    out = cc_k(y_true, y_pred)
#    out *= weight
#    return out.sum(axis=1).mean()

def compile_model(mdl):
    mdl.compile(
        optimizer=optimizers.Adadelta(0.7),
        #optimizer=optimizers.AdamW(
        #    learning_rate=3e-3,
        #),
        loss=#cce,
            losses.CategoricalCrossentropy(
                from_logits=True,
                name='cce',
            ),
        metrics=[
        ],
        #jit_compile=False,
    )

if __name__=='__main__':
    main()
