import numpy as np
import const
import data
import keras
from keras import layers, activations

@keras.saving.register_keras_serializable("Transformer")
class AttentionBlock(layers.Layer):
    def __init__(self, hdm, odm, **kwargs):
        super().__init__(**kwargs)
        self.hdm = hdm
        self.odm = odm

    def build(self, input_shape):
        ln = input_shape[1]
        idm = input_shape[2]
        hdm = self.hdm
        odm = self.odm
        
        self.k = self.add_weight(
            shape=(idm, hdm),
            initializer='random_normal',
            trainable=True,
        )
        self.q = self.add_weight(
            shape=(idm, hdm),
            initializer='random_normal',
            trainable=True,
        )
        self.v = self.add_weight(
            shape=(idm, idm),
            initializer='random_normal',
            trainable=True,
        )
        self.mlp = self.add_weight(
            shape=(idm, odm),
            initializer='random_normal',
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(ln, odm),
            initializer='random_normal',
            trainable=True,
        )
    
    def call(self, i):
        T = lambda x: keras.ops.transpose(x, axes=(0,2,1))
        keys = T(i @ self.k)
        queries = i @ self.q
        act = queries @ keys
        act = activations.softmax(act)
        vls = i @ self.v
        ret = (act @ vls) + i
        return activations.leaky_relu(ret @ self.mlp) + self.bias

@keras.saving.register_keras_serializable("Transformer")
class Extract(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, i):
        return i[:, -1, :]

@keras.saving.register_keras_serializable("mqe")
def mqe(y_true, y_pred):
    return ((10*(y_true-y_pred))**4).mean()

def gen_model():
    osz = data.mxcap-data.mncap+1
    model = keras.Sequential(
        [
            #layers.Dense(60, activation="leaky_relu"),
            #layers.Dense(40, activation="leaky_relu"),
            layers.Dense(200, activation="leaky_relu"),
            #layers.BatchNormalization(),
            
            AttentionBlock(20, 200),
            layers.BatchNormalization(),
            AttentionBlock(20, 150),
            layers.BatchNormalization(),
            AttentionBlock(20, 150),
            layers.BatchNormalization(),
            AttentionBlock(30, 250),
            layers.BatchNormalization(),
            AttentionBlock(30, 250),
            layers.BatchNormalization(),
            #AttentionBlock(30, 250),
            #layers.BatchNormalization(),
            #AttentionBlock(30, 250),
            #layers.BatchNormalization(),
            AttentionBlock(30, osz),
            layers.BatchNormalization(),
            #Transformer(data.rlens-1, embed_space, 5),
            #layers.BatchNormalization(),
            #Transformer(data.rlens-1, embed_space, 5),
            #layers.BatchNormalization(),
            
            Extract(),
            layers.Flatten(),
            layers.Softmax(),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adamax(
            learning_rate=keras.optimizers.schedules.ExponentialDecay(
                7e-4,
                20,
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
        ]
    )
    return model

mdnum = '06'
if __name__=='__main__':
    mdl = gen_model()
    print("MODEL GENERATED")

    mdl.fit(
        data.data_generator(),
        epochs=1,
    )
    mdl.save(f'saved-models/{mdnum}.keras')
