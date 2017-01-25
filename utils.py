# utils

import keras.backend as K
from backend import repeat
# dropout
def Dropout(state_before, dropout):
    return K.dropout(state_before, dropout, seed=20080524)


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def ReplicateLayer(x, n_times):
    # (n_times,)+x.shape
    return repeat(K.expand_dims(x, 0), n_times)





