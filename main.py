import const
import data
import model
import keras
from keras import optimizers, losses, metrics

#dset_conf = { "path": "roneneldan/TinyStories", }
dset_conf = {
    "path": "HuggingFaceFW/fineweb",
    "name": "CC-MAIN-2024-10",
    "split": "train",
    "streaming": True,
}
dset_conf_mini = {**dset_conf, "name": "sample-10BT"}

mdnum = '13'
fp = f'saved-models/{mdnum}.keras'
if __name__=='__main__':
    mdl = model.gen_model()
    mdl.compile(
        optimizer=optimizers.Adadelta(learning_rate=1.0),
        loss=losses.CategoricalCrossentropy(from_logits=True),
    )
    mdl.summary()
    
    mdl.fit(
        data.data_generator(dset_conf),
        epochs=100,
        steps_per_epoch=100,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                fp,
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch",
            ),
        ],
        validation_data=data.data_generator(dset_conf_mini),
        validation_steps=5,
    )
