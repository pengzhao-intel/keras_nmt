# initializations

import keras.backend as K
import numpy as np

def norm_weight(shape, scale=0.01, name=None):
    return K.random_normal_variable(shape, 0.0, scale, name=name)

def ortho_weight(shape, scale=1.0, name=None):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return K.variable(scale * q[:shape[0], :shape[1]], name=name)

def constant_weight(shape, value=0., name=None):
    return  K.variable(np.ones(shape) * value, name=name)


