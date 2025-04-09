seed = 1337

import os
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['KERAS_BACKEND'] = 'jax'
import random
random.seed(seed)
import numpy as np
np.random.seed(seed)
import keras
keras.utils.set_random_seed(seed)

temp = 1

batch = 32
rlens = 64 #128

mdnum = '14'
fp = f'saved-models/{mdnum}.keras'
tknum = '01'
fp_tk = f'saved-tokenizers/{tknum}.pickle'


#import jax
#from keras import distribution as D
#distro = D.DeviceMesh(
#    shape=(2,),
#    axis_names=['ax'],
#    #devices=dev,
#)
#layout = D.LayoutMap(distro)
#mp = D.ModelParallel(layout_map=layout)
#D.set_distribution(mp)
