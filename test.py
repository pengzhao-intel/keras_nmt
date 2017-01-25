'''
Created on Sep 25, 2016

@author: lxh5147
'''
import logging
logging.basicConfig()
logger = logging.getLogger('mt_exp')

import numpy as np
# to fix the maximum recursion depth exceeded while calling a Python object, refer to:https://github.com/Theano/Theano/issues/689
import sys; sys.setrecursionlimit(50000)
import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend as K

def lookup_table(table, indice, name=None):
    zero_mask = K.zeros_like(indice)
    one_mask = K.ones_like(indice)
    if K._BACKEND == 'tensorflow':
        from tensorflow import select as _select
    else:
        from theano.tensor import switch as _select
    mask = _select (indice < 0 , zero_mask, one_mask)
    indice *= mask
    output = K.gather(table, indice)
    output *= K.cast(K.expand_dims(mask), dtype=K.dtype(output))
    return output

def test_cross_entropy():
    from nmt import get_category_cross_entropy_from_flat_logits

    logits_flat = K.placeholder((None, 4))
    targets = K.placeholder((None,), dtype='int64')
    outputs = get_category_cross_entropy_from_flat_logits(logits_flat, targets)
    fn = K.function([logits_flat, targets], outputs=[outputs])
    logits_flat_val = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    targets_val = [0, 3]
    outputs_val = fn([logits_flat_val, targets_val])
    print outputs_val

def test_gradient():
    pass

def get_category_cross_entropy_from_flat_logits(logits_flat, targets, mask=None):
    targets_flat = K.flatten(targets)
    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, K.cast(targets_flat, 'int64'))
    else:
        # Theano will internally call onehot version if the two dims do not match
        ce = K.categorical_crossentropy(logits_flat, targets_flat, from_logits=True)
    if mask is not None:
        mask_flat = K.flatten(mask)
        ce *= mask_flat
        return K.sum(ce) / K.sum(mask_flat)
    else:
        return K.mean(ce)

def get_probs_from_logits(logits):
    logits_shape = K.shape(logits)
    logits_flat = K.reshape(logits, shape=(-1, logits_shape[K.ndim(logits) - 1]))
    probs_flat = K.softmax(logits_flat)
    return K.reshape(probs_flat, shape=logits_shape)


def max_gradient():
    x = K.variable(1.0)
    y = K.variable(1.0)
    z = K.maximum(x, y)
    grads = K.gradients(z, variables=[x, y])
    fn = K.function(inputs=[], outputs=grads)
    print 'grads'
    print fn([])



if __name__ == '__main__':
    test_cross_entropy()
    max_gradient()
    state_below = K.placeholder(shape=(3, 2, 4))
    print state_below[0, :, :]
    mask = K.ones_like(K.sum(state_below, axis=[1, 2], keepdims=True))
    print K.shape(mask)
    y = state_below * mask
    steps = K.shape(state_below)[0]
    w = K.reshape(state_below, shape=(steps, -1))
    print steps
    print w
    state_below[:2, :, :]
    print K.sum(state_below, axis=[0, 2])
    print [1] * 5
    ntimes = K.placeholder(shape=(2,), dtype='int32')
    # state_below[:, :, ntimes[0]]
    # K.gather(reference, indices)
    ntime = ntimes[0]

    print K.one_hot(K.cast(state_below, dtype='int32'), nb_classes=ntime)
    x = K.variable(0.0)
    print K.get_value(x)
    ite = K.placeholder(ndim=0)
    print ite
    print ite ** 2

    w = K.placeholder((6, 5))
    x = K.placeholder((None, None, 6))
    y = K.placeholder((None, None, 5))
    y_shape = K.shape(y)
    y = K.reshape(y, shape=(-1, 5))
    y2 = K.reshape(y, shape=y_shape)
    print y2
    # wx = K.dot(x, w)

    logits = K.placeholder((None, None, None))
    # print K.softmax(logits)

    print logits / K.cast(y_shape[0], dtype=K.dtype(logits))
    print K.switch(ite > 2 , ntimes[0], ntimes[1])

    y = K.placeholder((None,), dtype='int32')
    embedding = K.placeholder((None, None))
    z = lookup_table(embedding, y)

    f = K.function([y, embedding], [z])
    y_val = [-1, 0, 1]
    embedding_val = [[1, 1], [2, 2]]
    print (f([y_val, embedding_val])[0])

    x = K.placeholder((None, None))
    mask = K.placeholder((None, 1))
    y = x * K.expand_dims(K.squeeze(mask, axis=1))
    print (y)
    f = K.function([x, mask], [y])
    print f([[[1, 2], [3, 4]], [[1], [2]]])[0]


