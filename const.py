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
