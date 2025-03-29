import const
import data
import numpy as np
import keras
from keras import layers, activations, ops

inf = float('inf')

@keras.saving.register_keras_serializable("Attentionx1")
class Attention(layers.Layer):
    def __init__(self, hdm, odm, **kwargs):
        super().__init__(**kwargs)
        self.hdm = hdm
        self.odm = odm

    def build(self, input_shape):
        ln = self.ln = input_shape[1]
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
    
    def call(self, i):
        T = lambda x: keras.ops.transpose(x, axes=(0,2,1))
        keys = T(i @ self.k)
        queries = i @ self.q
        act = queries @ keys
        #for di in range(self.ln):
        #    act.at[:, di, di+1:].set(-inf)
        act = activations.softmax(act)
        vls = i @ self.v
        ret = (act @ vls) + i
        return ret

@keras.saving.register_keras_serializable("AttentionxN")
class MHAttention(layers.Layer):
    def __init__(self, hdm, odm, heads, **kwargs):
        super().__init__(**kwargs)
        self.heads = [
            Attention(hdm, odm)
            for i in range(heads)
        ]
    
    def build(self, input_shape):
        for h in self.heads:
            h.build(input_shape)
    
    def call(self, i):
        return ops.concatenate([ly(i) for ly in self.heads], axis=-1)

@keras.saving.register_keras_serializable("PosEncode")
class PosEncode(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        ln = input_shape[1]
        self.id = np.array([
            [
                [
                    (row==col)
                    for col in range(ln)
                ]
                for row in range(ln)
            ]
            for ly in range(input_shape[0])
        ])
    
    def call(self, i):
        return ops.concatenate((i, self.id[:i.shape[0], :, :]), axis=-1)

def gen_model():
    osz = data.mxcap-data.mncap+1
    model = keras.Sequential(
        [
            PosEncode(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            
            MHAttention(10, 500, 4),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            MHAttention(10, 500, 4),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            MHAttention(10, 500, 4),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            layers.Dense(500, activation="leaky_relu"),
            layers.BatchNormalization(),
            
            layers.Dense(data.span, activation="leaky_relu"),
            layers.BatchNormalization(),
            layers.Softmax(),
            layers.BatchNormalization(),
        ]
    )
    return model
