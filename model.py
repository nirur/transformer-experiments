import const
import data
import numpy as np
import keras
from keras import layers, activations, ops

inf = float('inf')

@keras.saving.register_keras_serializable()
class Attention(layers.Layer):
    def __init__(self, hdm, **kwargs):
        super().__init__(**kwargs)
        self.hdm = hdm

    def build(self, input_shape):
        ln = self.ln = input_shape[1]
        idm = input_shape[2]
        hdm = self.hdm
        
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
        self.v1 = self.add_weight(
            shape=(idm, hdm),
            initializer='random_normal',
            trainable=True,
        )
        self.v2 = self.add_weight(
            shape=(hdm, idm),
            initializer='random_normal',
            trainable=True,
        )
        self.subs = np.zeros((ln,ln))
        for i in range(ln):
            self.subs[i, i+1:] = -inf
    
    def call(self, i):
        T = lambda x: keras.ops.transpose(x, axes=(0,2,1))
        keys = T(i @ self.k)
        queries = i @ self.q
        act = queries @ keys
        act = (act + self.subs) / self.hdm**0.5
        act = activations.softmax(act)
        vls = i @ self.v1 @ self.v2
        return act @ vls

@keras.saving.register_keras_serializable()
class MHAttention(layers.Layer):
    def __init__(self, hdm, heads, **kwargs):
        super().__init__(**kwargs)
        self.heads = [
            Attention(hdm)
            for i in range(heads)
        ]
    
    def build(self, input_shape):
        for h in self.heads:
            h.build(input_shape)
    
    def call(self, i):
        return sum([ly(i) for ly in self.heads])

@keras.saving.register_keras_serializable()
class PosEncode(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        ln = input_shape[1]
        self.id = np.eye(ln)[None, :, :]
    
    def call(self, i):
        return ops.concatenate((i, np.repeat(self.id, i.shape[0], axis=0)), axis=-1)

def gen_model():
    enc_dim = 64
    m = inp = keras.Input(shape=(data.rlens, data.span))
    
    m = PosEncode()(m)
    m = layers.Dense(enc_dim, use_bias=False)(m)
    
    for i in range(4):
        m = m + MHAttention(16, 4)(m)
        m2 = layers.Dense(256, activation="relu")(m)
        m2 = layers.Dense(enc_dim)(m2)
        m = m + m2
        m = layers.LayerNormalization()(m)
        m = layers.LayerNormalization()(m)
    
    m = layers.LayerNormalization()(m)
    m = layers.Dense(data.span)(m)
    #m = layers.BatchNormalization()(m)
    #m = layers.Softmax()(m)
    #m = layers.BatchNormalization()(m)
    
    model = keras.Model(inputs=inp, outputs=m)
    model.summary()
    return model
