import const
import data
import model
import keras
from keras import layers, activations

@keras.saving.register_keras_serializable("mqe")
def mqe(y_true, y_pred):
    return ((10*(y_true-y_pred))**4).mean()

def compile_model(mdl):
    mdl.compile(
        optimizer=keras.optimizers.Adamax(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                3e-4,
                1e5,
                0.8,
                False,
            ),
        ),
        loss=
            mqe,
        metrics=[
            'categorical_accuracy',
            'false_negatives',
            'mse',
            'kl_divergence',
        ],
    )

mdnum = '06'
fp = f'saved-models/{mdnum}.keras'
if __name__=='__main__':
    mdl = model.gen_model()
    compile_model(mdl)
    print("MODEL GENERATED")

    mdl.fit(
        data.data_generator(),
        epochs=1,
        steps_per_epoch=data.samples,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                fp,
                monitor="loss",
                save_best_only=True,
                save_freq=10_000,
            ),
        ],
    )
    mdl.save(fp)
