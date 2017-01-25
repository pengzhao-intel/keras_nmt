# learning algorithms
import keras.backend as K
from backend import clip_norm
from itertools import izip

def adadelta(parameters, gradients, rho=0.95, eps=1e-8, with_fast_training=False, fast_training_parameters=[], base_training_ratio=0.1):
    # create variables to store intermediate updates
    shapes = [K.get_variable_shape(p) for p in parameters]
    gradients_sq = [K.zeros(shape) for shape in shapes]
    deltas_sq = [K.zeros(shape) for shape in shapes]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho * g_sq + (1 - rho) * K.square(g)
    				   for g_sq, g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    # sometimes it will output nan for Euclidean distance cost
    deltas = [(K.sqrt(d_sq + eps) / K.sqrt(g_sq + eps)) * grad
    		 for d_sq, g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho * d_sq + (1 - rho) * K.square(d) for d_sq, d in izip(deltas_sq, deltas)]

    # Prepare it as a list f
    gradient_sq_updates = [K.update(p, new_p) for (p, new_p) in  zip(gradients_sq, gradients_sq_new)]
    deltas_sq_updates = [K.update(p, new_p) for (p, new_p) in zip(deltas_sq, deltas_sq_new)]
    # parameters_updates = [(p,T.clip(p - d, -15,15)) for p,d in izip(parameters,deltas)]
    if with_fast_training:
        # for fast training, we update the new parameters with normal delta (i.e., 1.0),
        # but update the original parameters with small delta (e.g., 0.1 or some dynamic ratio)
        parameters_updates = []

        for p, d in izip(parameters, deltas):
            if p.name in fast_training_parameters:
                parameters_updates.append(K.update(p, p - d))
            else:
                parameters_updates.append(K.update(p, p - base_training_ratio * d))
    else:
        parameters_updates = [K.update(p, p - d) for p, d in izip(parameters, deltas)]
        '''
        parameters_updates = []
        for p,d in izip(parameters,deltas):
            if p.name == 'rnn_inverse_decoder_W_ch':
                new_p = theano.printing.Print('rnn_inverse_decoder_W_ch:')(p)
                new_d = theano.printing.Print('delta:')(d)
                parameters_updates.append((p,new_p-new_d))
            else:
                parameters_updates.append((p,p-d))
        '''
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


def grad_clip(grads, clip_c, eps=1e-6):
    # apply gradient clipping
    if clip_c > 0:
        g2 = 0.
        for g in grads:
            g2 += K.sum(K.square(g))
        n = K.sqrt(g2 + eps)
        new_grads = []
        for g in grads:
            new_grads.append(clip_norm(g, clip_c, n))
        grads = new_grads

    return grads
