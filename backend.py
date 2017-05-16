'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K
from keras.backend.common import _FLOATX
import numpy as np


def repeat(x, n):
    '''Repeats a tensor along the first dimension:
    For example, if x has shape (samples, dim) and n=2,
    the output will have shape (samples*2, dim)

    # Parameters
    ----------
    x : a tensor
    n: times to repeat

    # Returns
    ------
    the repeated tensor

    '''
    x_shape = K.shape(x)
    x_ndim = K.ndim(x)
    # to 1D tensor
    x_tiled = K.tile(K.reshape(x, (-1,)), n)
    # re-shape to (n,...)
    x_tiled_shape = pack([n] + [x_shape[i] for i in range(x_ndim)])
    output = K.reshape(x_tiled, x_tiled_shape)
    pattern = [1, 0] + [i + 1 for i in range(1, x_ndim)]
    output = K.permute_dimensions(output, pattern)
    output_shape = pack([n * x_shape[0]] + [x_shape[i] for i in range(1, x_ndim)])
    return K.reshape(output, output_shape)


def pack(tensor_list):
    output = K.stack(tensor_list)
    output.num = len(tensor_list)
    return output

if K._BACKEND == 'theano':
    import theano
    from theano import tensor as T

    def dot(x, y):
        return T.dot(x, y)


    def clip_norm(g, c, n):
        if c > 0:
            g = K.switch(n >= c, g * c / n, g)
        return g

    def shift_right(x):
        '''Gets one right shifted along time dimension of x, padding with zeros
        # Parameters
        ----------
        x : a tensor of shape nb_samples, time_steps, input_dim
 
        # Returns
        ------
        One right shifted tensor
        '''
        y = K.zeros_like(x)
        return T.set_subtensor(y[:, 1:, :], x[:, :-1, :])

    def foreach(x, step_func, dtype=None, name=None):
        '''Process each element in x and returns all the processed outputs in a tensor.
        # Parameters
        ----------
        x : a tensor
        step_func: a function that process an element of the input tensor and output a new tensor, e.g., lambda xi: xi+2.
        dtype: dtype of the output tensor. By default output tensor has the same dtype as x
        # Returns
        ------
        A tensor that packs all the outputs.
        '''
        return theano.scan(fn=step_func, sequences=[x], name=name)[0]

    def scan(fn, sequences, outputs_initials, name=None):
        '''Process multiple sequences, and returns a list of tensors. Each output tensor list corresponds to one tensor in the outputs_initials.
        # Parameters
        ----------
        sequences : a list of tensors
        fn: a function that process previous output tensors and current input tensors, and returns current output tensors
        outputs_initials: initial output tensors
        name: name of the returned tensor
        # Returns
        ------
        A list of output tensors.
        '''
        # warning: updates dictionary ignored
        return theano.scan(fn, sequences=sequences, outputs_info=outputs_initials, name=name)[0]

    def random_multinomial(n=1, pvals=None, dtype=_FLOATX, seed=None):
        if seed is None:
            seed = np.random.randint(1, 10e6)
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=n, pvals=pvals, dtype=dtype)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf

    # support None
    def dot(x, y):
        '''Multiplies 2 tensors.
        When attempting to multiply a ND tensor
        with a ND tensor, reproduces the Theano behavior
        (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
        '''
        ndim_x = K.ndim(x)
        ndim_y = K.ndim(y)

        if ndim_x is not None and ndim_x > 2 or ndim_y > 2:
            x_shape = tf.shape(x)
            y_shape = tf.shape(y)
            y_permute_dim = list(range(ndim_y))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, pack([-1, x_shape[ndim_x - 1]]))
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), pack([y_shape[ndim_y - 2], -1]))
            target_shape = [x_shape[i] for i in range(ndim_x - 1)] + [y_shape[i] for i in range(ndim_y - 2)] + [y_shape[ndim_y - 1]]
            return tf.reshape(tf.matmul(xt, yt), pack(target_shape))
        out = tf.matmul(x, y)
        return out

    def clip_norm(g, c, n):
        if c > 0:
            from tensorflow.python.ops import control_flow_ops

            f = control_flow_ops.cond(tf.cast(n >= c, 'bool'),
                                        lambda: c / n,
                                        lambda: tf.constant(1.0))
        return tf.scalar_mul(f, g)

    def shift_right(x):
        '''Gets one right shifted along time dimension of x, padding with zeros
        # Parameters
        ----------
        x : a tensor of shape nb_samples, time_steps, input_dim

        # Returns
        ------
        One right shifted tensor
        '''
        last_removed = K.reverse(K.reverse(x, axes=1)[:, 1:, :], axes=1)
        padding = K.expand_dims(K.zeros_like(x[:, 0, :]), axis=1)
        return K.concatenate([padding, last_removed], axis=1)

    def foreach(x, step_func, dtype=None, name=None):
        '''Process each element in x and returns all the processed outputs in a tensor. 
        # Parameters
        ----------
        x : a tensor
        step_func: a function that process an element of the input tensor and output a new tensor, e.g., lambda xi: xi+2.
        dtype: dtype of the output tensor. By default output tensor has the same dtype as x
        # Returns
        ------
        A tensor that packs all the outputs.
        '''
        from tensorflow.python.ops import tensor_array_ops
        size = K.shape(x)[0]
        accs_ta = tensor_array_ops.TensorArray(dtype=dtype if dtype else x.dtype,
                                                  size=size,
                                                  dynamic_size=False,
                                                  infer_shape=True)
        i = tf.constant(0)
        def b(i, tas):
            output = step_func(K.gather(x, i))
            tas = tas.write(i, output)
            return (i + 1, tas)

        _1, outputs = tf.while_loop(lambda i, _: i < size, b, [i, accs_ta])
        return outputs.pack(name=name)

    def scan(fn, sequences, outputs_initials, name=None):
        '''Process multiple sequences, and returns a list of tensors. Each output tensor list corresponds to one tensor in the outputs_initials.
        # Parameters
        ----------
        sequences : a list of tensors
        fn: a function that process previous output tensors and current input tensors, and returns current output tensors
        outputs_initials: initial output tensors
        name: name of the returned tensor
        # Returns
        ------
        A list of output tensors.
        '''
        return tf.scan(fn, elems=sequences, initializer=outputs_initials, name=name)

    def random_multinomial(n=1, pvals=None, dtype=_FLOATX, seed=None):
        samples = tf.multinomial(tf.log(pvals), num_samples=n, seed=seed)
        # one_hot: batch_size, n, nb_classes -> sum: batch_size, nb_classes
        samples = K.sum (tf.one_hot(samples, K.shape(pvals)[K.ndim(pvals) - 1]), axis=-2)
        if dtype and not  dtype == K.dtype(samples):
            samples = K.cast(samples, dtype)
        return samples
