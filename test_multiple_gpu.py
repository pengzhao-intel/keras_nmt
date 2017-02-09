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
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "theano"
import tensorflow as tf
# with tf.device('/cpu:0'):
#    import keras.backend as K

import keras.backend as K
session = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
K.set_session(session)
# conclusion: put all variables/model into the parameter server

def pred(x, W):
    return K.dot(x, W)

def main():
    inputs_x = []
    pred_list = []
    W = K.variable(np.zeros((2, 3)), name='W')
    for device in devices:
        x = K.placeholder((None, 2), 'x')

        inputs_x.append(x)

        with tf.device(device):
            l = pred(x, W)
            pred_list.append(l)

    with tf.device('/cpu:0'):
        fn = K.function(inputs_x , pred_list)
        inputs_x_val = [ [[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    print fn(inputs_x_val)

devices = ['/gpu:0', '/gpu:1']


def main_model_parallel():


    W = tf.Variable(np.zeros((2, 4)), dtype=tf.float32, name='W')
    x = tf.placeholder(tf.float32, (None, 2), 'x')

    Ws = tf.split(1, 2, W)
    # with tf.device(devices[0]):

    ls = []
    for w, device in zip(Ws, devices):
        with tf.device(device):
            l = pred(x, w)
            ls.append(l)

    L = K.concatenate(ls)

    with tf.device(devices[1]):
        L2 = L + 1

    # initialize the model
    with tf.device(devices[0]):
        session.run(tf.initialize_all_variables())

    fn = K.function([x], [ L, L2])
    inputs_x_val = [ [[1, 1], [1, 1]] ]    # x: 2,2
    print fn(inputs_x_val)

if __name__ == '__main__':

    with tf.device('/cpu:0'):
        main_model_parallel()


