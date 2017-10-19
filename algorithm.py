import keras.backend as K
from backend import clip_norm
from itertools import izip
from mlsl_api import AllReduce, addr

def adadelta(mlsl_obj, dist, parameters, gradients, rho=0.95, eps=1e-6):
    # create variables to store intermediate updates
    if mlsl_obj != None:
        reduce_op=AllReduce(mlsl_obj,dist,1)

    shapes = [K.get_variable_shape(p) for p in parameters]
    gradients_sq = [K.zeros(shape) for shape in shapes]
    deltas_sq = [K.zeros(shape) for shape in shapes]
    
    if mlsl_obj != None:
    # collect grad first
        gradients = [reduce_op(grad) for grad in gradients]    

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho * g_sq + (1 - rho) * K.square(g)
                       for g_sq, g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [(K.sqrt(d_sq + eps) / K.sqrt(g_sq + eps)) * grad
             for d_sq, g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho * d_sq + (1 - rho) * K.square(d) for d_sq, d in izip(deltas_sq, deltas)]

    gradient_sq_updates = [K.update(p, new_p) for (p, new_p) in  zip(gradients_sq, gradients_sq_new)]
    deltas_sq_updates = [K.update(p, new_p) for (p, new_p) in zip(deltas_sq, deltas_sq_new)]

    parameters_updates = [K.update(p, p - d) for p, d in izip(parameters, deltas)]

    return gradient_sq_updates + deltas_sq_updates + parameters_updates

def grad_clip(grads, clip_c):
    # apply gradient clipping
    if clip_c > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, clip_c, norm) for g in grads]
    return grads

