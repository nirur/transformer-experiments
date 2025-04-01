import const
import data
import model
import keras
from keras import optimizers, losses, metrics

#@keras.saving.register_keras_serializable()
#def mqe(y_true, y_pred):
#    return ((10*(y_true-y_pred))**4).mean()

def compile_model(mdl):
    mdl.compile(
        optimizer=optimizers.Adadelta(
            learning_rate=1.0,
            #epsilon=1e-8,
            #learning_rate=keras.optimizers.schedules.ExponentialDecay(
            #    3e-4,
            #    1e2,
            #    0.8,
            #    False,
            #),
        ),
        loss=losses.CategoricalCrossentropy(
            from_logits=True,
            reduction="mean",
        ),
        metrics=[
            #'categorical_accuracy',
            #'false_negatives',
            #'kl_divergence',
            #mqe,
            #tce,
        ],
    )

mdnum = '10'
fp = f'saved-models/{mdnum}.keras'
if __name__=='__main__':
    mdl = model.gen_model()
    compile_model(mdl)
    print("MODEL GENERATED")

    mdl.fit(
        data.file_generator(),
        epochs=100,
        steps_per_epoch=500,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                fp,
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch",
            ),
        ],
        validation_data=next(iter(
            data.file_generator(split="test")
        )),
    )
