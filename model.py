import const
import data
import numpy as np
import keras
from keras import \
    layers as L, activations as A, ops as O, initializers as I

def gen_model():
    enc_dim = 256
    m = inp = keras.Input(shape=(const.rlens, data.span))
    
    tied_encode = PosEncode(enc_dim)
    m = tied_encode(m)
    
    ly = 4
    for i in range(1,ly+1):
        m1 = m
        m1 = MHAttn(8, 1, ly**-0.5)(m1)
        m1 = L.LayerNormalization()(m1)
        if True:#i%4:
            m += m1
        else:
            m1 = A.tanh(m1)
            m1 = L.LayerNormalization()(m1)
            m *= 1 + m1
        #m = L.LayerNormalization()(m)
        
        m2 = m
        m2 = FFW(enc_dim, 4)(m2)
        m2 = L.LayerNormalization()(m2)
        m += m2
    
    #unembed = O.transpose(tied_encode.embedding) # tie to encoding
    #m = m @ unembed
    m = L.Dense(data.span)(m)
    m = L.LayerNormalization()(m)
    
    model = keras.Model(inputs=inp, outputs=m)
    return model

inf = float('inf')

@keras.saving.register_keras_serializable()
class PosEncode(L.Layer):
    def __init__(self, enc_dim, **kwargs):
        super().__init__(**kwargs)
        self.enc_dim = enc_dim
    
    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(input_shape[-1], self.enc_dim),
            initializer='ones',
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(input_shape[1], self.enc_dim),
            initializer='zeros',
            trainable=True,
        )
    
    def call(self, i):
        return (i @ self.embedding) + self.bias

@keras.saving.register_keras_serializable()
class MHAttn(L.Layer):
    def __init__(self, heads, shrink, init_dev, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.init_dev = init_dev
        self.shrink = shrink
    
    def build(self, input_shape):
        ln = input_shape[1]
        idm = input_shape[2]
        heads = self.heads
        hdm = self.hdm = idm // (heads*self.shrink)
        
        self.k = self.add_weight(
            shape=(heads, idm, hdm),
            initializer='random_normal',
            trainable=True,
        )
        self.q = self.add_weight(
            shape=(heads, idm, hdm),
            initializer='random_normal',
            trainable=True,
        )
        self.v = self.add_weight(
            shape=(heads, idm, idm),#hdm*self.shrink),
            initializer=I.RandomNormal(stddev=self.init_dev),
            trainable=True,
        )
        #self.v2 = self.add_weight(
        #    shape=(hdm*self.shrink, idm),
        #    initializer=I.RandomNormal(stddev=self.init_dev),
        #    trainable=True,
        #)
        self.subs = np.zeros((ln,ln))
        for i in range(ln):
            self.subs[i, i+1:] = -inf
        self.bias = self.add_weight(
            shape=(idm,),
            initializer=I.RandomNormal(stddev=self.init_dev),
            trainable=True,
        )
    
    def call(self, i):
        T = lambda x: O.transpose(x, axes=(0,1,3,2))
        i = i[:, None, :, :]
        i = O.repeat(i, self.heads, axis=1)
        keys = T(i @ self.k)
        queries = i @ self.q
        act = queries @ keys
        act = (act + self.subs) / self.hdm**0.5
        act = A.softmax(act)
        vls = i @ self.v #@ self.v2
        deltas = (act @ vls).sum(axis=1)
        return deltas + self.bias

@keras.saving.register_keras_serializable()
class FFW(L.Layer):
    def __init__(self, orig, scale, **kwargs):
        super().__init__(**kwargs)
        self.orig = orig
        self.scale = scale
        
        self.d1 = L.Dense(orig*scale)
        self.d2 = L.Dense(orig)
        self.dp = L.Dropout(0.1)
    
    def build(self, i_s):
        i_s = list(i_s)
        self.d1.build(i_s)
        i_s[-1] *= self.scale
        self.d2.build(i_s)
    
    def call(self, i):
        i = self.d1(i)
        i = A.leaky_relu(i)
        i = self.d2(i)
        #i = self.dp(i)
        return i
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "orig": self.orig,
            "scale": self.scale,
        })
        return config
