# models: BidirectionalEncoder, Decoder, GRU, LogisticRegression, LookupTable
import numpy
from initialization import constant_weight, ortho_weight, norm_weight

from utils import ReplicateLayer, _p
import keras.backend as K

import backend
import mkl_gru

K.scan = backend.scan

class MKL_GRU(object):

    def __init__(self, n_in, n_hids, name='GRU', with_context=False, max_len=None):

        self.n_in = n_in
        self.n_hids = n_hids
        self.pname = name
        self.max_len = max_len

        self.with_context = with_context
        if self.with_context:
            self.c_hids = n_hids

        self._init_params()


    def _init_params(self):

        shape_xh = (self.n_in*3, self.n_hids)
        shape_hh = (self.n_hids*3, self.n_hids)
        self.W_x = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_x'))
        self.b = constant_weight(shape=(self.n_hids*3,), name=_p(self.pname, 'b'))
        self.W_h = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_h'))
        self.params = [self.W_x, self.W_h, self.b]
        self.GRU_op = mkl_gru.GRU(hid=self.n_hids, return_sequences=True, max_len=self.max_len)
        self.h_init_state = numpy.zeros((80, 1000), numpy.float64)

    def apply(self, state_below, mask_below=None, init_state=None, context=None):

        if K.ndim(state_below) == 2:
            state_below = K.expand_dims(state_below, 1)

        if mask_below is None:
            mask_below = K.ones_like(K.sum(state_below, axis=2, keepdims=True))

        if K.ndim(mask_below) == 2:
            mask_below = K.expand_dims(mask_below)

        #if init_state is None:
            # nb_samples,n_hids
        init_state = K.repeat_elements(K.expand_dims(K.zeros_like(K.sum(state_below, axis=[0, 2]))), self.n_hids, axis=1)
        '''
        state_below_xh = K. dot(state_below, self.W_xh)
        state_below_xz = K. dot(state_below, self.W_xz)
        state_below_xr = K.dot(state_below, self.W_xr)
        sequences = [state_below_xh, state_below_xz, state_below_xr, mask_below]
        fn = lambda x_h, x_z, x_r, x_m, h_tm1: self._step(x_h, x_z, x_r, x_m, h_tm1)
        rval = K.scan(fn, sequences=sequences, outputs_initials=init_state, name=_p(self.pname, 'layers'))
        self.output = rval
        '''
        #print('1,', K.ndim(init_state))
        #print(init_state.shape)
        #exit()
        self.output = self.GRU_op(state_below, self.W_x, self.W_h, init_state, self.b)[0]
        #print('2,', K.ndim(self.output))
        return self.output



class BidirectionalEncoder(object):

    def __init__(self, n_in, n_hids, table, mkl, name='rnn_encoder', max_len=None):

        # lookup table
        self.table = table
        # embedding dimension
        self.n_in = n_in
        # hidden state dimension
        self.n_hids = n_hids
        self.mkl = mkl
        self.params = []
        self.layers = []
        self.max_len = max_len

        if self.mkl == True:
            print('with mkl')
            self.forward = MKL_GRU(self.n_in, self.n_hids, name=_p(name, 'forward'), max_len=max_len)
        else:
            print('with no mkl')
            self.forward = GRU(self.n_in, self.n_hids, name=_p(name, 'forward'))
        self.layers.append(self.forward)
        if self.mkl == True:
            self.backward = MKL_GRU(self.n_in, self.n_hids, name=_p(name, 'backward'), max_len=max_len)
        else:
            self.backward = GRU(self.n_in, self.n_hids, name=_p(name, 'backward'))
        self.layers.append(self.backward)

        for layer in self.layers:
            self.params.extend(layer.params)


    def apply(self, sentence, sentence_mask):

        state_below = self.table.apply(sentence)
        #print('bin ,',K.ndim(state_below))
        # make sure state_below: n_steps * batch_size * embedding
        if K.ndim(state_below) == 2:
            state_below = K.expand_dims(state_below, 1)
        #print('applying')
        hiddens_forward = self.forward.apply(state_below, sentence_mask)
        #print('encoder apply h_f dim ', K.ndim(hiddens_forward))
        if sentence_mask is None:
            hiddens_backward = self.backward.apply(K.reverse(state_below, axes=0))
        else:
            hiddens_backward = self.backward.apply(K.reverse(state_below, axes=0), K.reverse(sentence_mask, axes=0))

        training_c_components = []
        training_c_components.append(hiddens_forward)
        training_c_components.append(K.reverse(hiddens_backward, axes=0))

        # annotaitons = T.concatenate(training_c_components, axis=2)
        annotaitons = K.concatenate(training_c_components)

        return annotaitons

class Attention_(object):
    def __init__(self, n_hids, with_attention=True, name='Attention'):

        self.n_hids = n_hids
        self.pname = name
        self.with_attention = with_attention
        self._init_params()

    def _init_params(self):
        shape_hh = (self.n_hids, self.n_hids)
        if self.with_attention:
            self.B_hp = norm_weight(shape=shape_hh, name=_p(self.pname, 'B_hp'))
            self.b_tt = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_tt'))
            self.D_pe = norm_weight(shape=(self.n_hids, 1), name=_p(self.pname, 'D_pe'))

            self.params = [self.B_hp, self.b_tt, self.D_pe]

    def apply(self, h1, c, p_from_c):
        h_shape = K.shape(h1)
        # time_stpes, 1, nb_samples, dim
        p_from_h = K.expand_dims(K.dot(h1, self.B_hp) + self.b_tt, axis=1)
        # 1, time_stpes, nb_samples, dim
        p_from_c = K.expand_dims(p_from_c, axis=0).repeat(h_shape[0], axis=0)
        p = p_from_h + p_from_c
        # energy = exp(dot(tanh(p), self.D_pe) + self.c_tt).reshape((source_len, target_num))
        # since self.c_tt has nothing to do with the probs, why? since it contributes an e^c_tt() to the the denominator and nominator
        # note: self.D_pe has a shape of (hidden_output_dim,1)
        # time_steps, nb_samples, 1
        energy = K.exp(K.dot(K.tanh(p), self.D_pe))

        normalizer = K.sum(energy, axis=1, keepdims=True)
        probs = energy / normalizer
        probs = K.squeeze(probs, axis=3)

        c = K.expand_dims(c, axis=0).repeat(h_shape[0], axis=0) 
        ctx = K.sum(c * K.expand_dims(probs), axis=1)
        return [ctx, probs]

class Decoder(object):

    def __init__(self, mkl, n_in, n_hids, n_cdim, maxout_part=2,
                 name='rnn_decoder',
                 with_attention=True,
                 with_coverage=False,
                 coverage_dim=1,
                 coverage_type='linguistic',
                 max_fertility=2,
                 with_context_gate=False,
                 max_len=None):

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_cdim = n_cdim
        self.maxout_part = maxout_part
        self.pname = name
        self.with_attention = with_attention
        self.with_coverage = with_coverage
        self.coverage_dim = coverage_dim
        assert coverage_type in ['linguistic', 'neural'], 'Coverage type must be either linguistic or neural'
        self.coverage_type = coverage_type
        self.max_fertility = max_fertility
        self.with_context_gate = with_context_gate
        self.mkl = mkl
        self.max_len = max_len
        self._init_params()
        #mkl decoder
        self.attention_ = Attention_(self.n_hids, name=_p(name, '_attention'))
        self.GRU_op = mkl_gru.GRU(hid=self.n_hids, return_sequences=True, max_len=self.max_len)

    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)

        self.W_xz = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xz'))
        self.W_xr = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xr'))
        self.W_xh = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_h'))
        self.W_hz = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hz'))
        self.W_hr = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hr'))
        self.W_hh = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        shape_ch = (self.n_cdim, self.n_hids)
        self.W_cz = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cz'))
        self.W_cr = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cr'))
        self.W_ch = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_ch'))
        self.W_c_init = norm_weight(shape=(self.n_cdim, self.n_hids), name=_p(self.pname, 'W_c_init'))
        self.b_c_init = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_c_init'))

        self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init, self.b_c_init]

        # we moved the parameters below here, to make it works for both with_context and with_attention modes
        # modification in this version
        # in the paper, e_{i,j} = a(s_{i-1}, h_j)
        # here, e_{i,j} = a(GRU(s_{i-1}, y_{i-1}), h_j), which considers the lastly generated target word
        # all the following parameters are for the introduced GRU
        # it is reasonable
        self.W_n1_h = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_h'))
        self.W_n1_r = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_r'))
        self.W_n1_z = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_z'))
        self.b_n1_h = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_h'))
        self.b_n1_r = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_r'))
        self.b_n1_z = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_z'))
        self.params += [self.W_n1_h, self.W_n1_r, self.W_n1_z, self.b_n1_h, self.b_n1_r, self.b_n1_z]

        if self.with_attention:
            self.A_cp = norm_weight(shape=shape_ch, name=_p(self.pname, 'A_cp'))
            self.B_hp = norm_weight(shape=shape_hh, name=_p(self.pname, 'B_hp'))
            self.b_tt = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_tt'))
            self.D_pe = norm_weight(shape=(self.n_hids, 1), name=_p(self.pname, 'D_pe'))

            self.params += [self.A_cp, self.B_hp, self.b_tt, self.D_pe]


            # coverage only works for attention model
            if self.with_coverage:
                shape_covh = (self.coverage_dim, self.n_hids)
                self.C_covp = norm_weight(shape=shape_covh, name=_p(self.pname, 'Cov_covp'))

                if self.coverage_type is 'linguistic':
                    # for linguistic coverage, fertility model is necessary since it yields better translation and alignment quality
                    self.W_cov_fertility = norm_weight(shape=(self.n_cdim, 1), name=_p(self.pname, 'W_cov_fertility'))
                    self.b_cov_fertility = constant_weight(shape=(1,), name=_p(self.pname, 'b_cov_fertility'))
                    self.params += [self.W_cov_fertility, self.b_cov_fertility]
                else:
                    # for neural network based coverage, gating is necessary
                    shape_covcov = (self.coverage_dim, self.coverage_dim)
                    self.W_cov_h = ortho_weight(shape=shape_covcov, name=_p(self.pname, 'W_cov_h'))
                    self.W_cov_r = ortho_weight(shape=shape_covcov, name=_p(self.pname, 'W_cov_r'))
                    self.W_cov_z = ortho_weight(shape=shape_covcov, name=_p(self.pname, 'W_cov_z'))
                    self.b_cov_h = constant_weight(shape=(self.coverage_dim,), name=_p(self.pname, 'b_cov_h'))
                    self.b_cov_r = constant_weight(shape=(self.coverage_dim,), name=_p(self.pname, 'b_cov_r'))
                    self.b_cov_z = constant_weight(shape=(self.coverage_dim,), name=_p(self.pname, 'b_cov_z'))

                    self.params += [self.W_cov_h, self.W_cov_r, self.W_cov_z, self.b_cov_h, self.b_cov_r, self.b_cov_z]

                    # parameters for coverage inputs
                    # attention probablity
                    self.W_cov_ph = norm_weight(shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_ph'))
                    self.W_cov_pr = norm_weight(shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_pr'))
                    self.W_cov_pz = norm_weight(shape=(1, self.coverage_dim), name=_p(self.pname, 'W_cov_pz'))
                    # source annotations
                    self.W_cov_ch = norm_weight(shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_ch'))
                    self.W_cov_cr = norm_weight(shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_cr'))
                    self.W_cov_cz = norm_weight(shape=(self.n_cdim, self.coverage_dim), name=_p(self.pname, 'W_cov_cz'))
                    # previous decoding states
                    self.W_cov_hh = norm_weight(shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hh'))
                    self.W_cov_hr = norm_weight(shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hr'))
                    self.W_cov_hz = norm_weight(shape=(self.n_hids, self.coverage_dim), name=_p(self.pname, 'W_cov_hz'))

                    self.params += [self.W_cov_ph, self.W_cov_pr, self.W_cov_pz, self.W_cov_ch, self.W_cov_cr, self.W_cov_cz, self.W_cov_hh, self.W_cov_hr, self.W_cov_hz]


        # for context gate, which works for both with_attention and with_context modes
        if self.with_context_gate:
            # parameters for coverage inputs
            # input form target context
            self.W_ctx_h = norm_weight(shape=(self.n_hids, self.n_hids), name=_p(self.pname, 'W_ctx_h'))
            self.W_ctx_c = norm_weight(shape=(self.n_cdim, self.n_hids), name=_p(self.pname, 'W_ctx_c'))
            self.b_ctx = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_ctx'))
            self.params += [self.W_ctx_h, self.W_ctx_c]

        # for readout
        n_out = self.n_in * self.maxout_part
        self.W_o_c = norm_weight(shape=(self.n_cdim, n_out), name=_p(self.pname, 'W_out_c'))
        self.W_o_h = norm_weight(shape=(self.n_hids, n_out), name=_p(self.pname, 'W_out_h'))
        self.W_o_e = norm_weight(shape=(self.n_in, n_out), name=_p(self.pname, 'W_out_e'))
        self.b_o = constant_weight(shape=(n_out,), name=_p(self.pname, 'b_out_o'))

        self.params += [self.W_o_c, self.W_o_h, self.W_o_e, self.b_o]



    # here we don't have parameters for coverage model, but with the fixed-length source context vector
    def _step_context(self, x_h, x_z, x_r, x_m, h_tm1, cz, cr, ch, ctx):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''

        # here h1 combines previous hidden state and lastly generated word with GRU
        # note that this is different from the paper
        z1 = K.sigmoid(K.dot(h_tm1, self.W_n1_z) + x_z + self.b_n1_z)
        r1 = K.sigmoid(K.dot(h_tm1, self.W_n1_r) + x_r + self.b_n1_r)
        h1 = K.tanh(r1 * K.dot(h_tm1, self.W_n1_h) + x_h + self.b_n1_h)
        h1 = z1 * h_tm1 + (1. - z1) * h1
        h1 = x_m * h1 + (1. - x_m) * h_tm1



        if self.with_context_gate:
            gate = K.sigmoid(K.dot(h1, self.W_ctx_h) +
                                  K.dot(ctx, self.W_ctx_c) + self.b_ctx)

            # we directly scale h1, since it used in computing both can_h_t and h_t
            h1 = h1 * (1. - gate)
        else:
            gate = 1.


        z_t = K.sigmoid(K.dot(h1, self.W_hz) + gate * cz + self.b_z)
        r_t = K.sigmoid(K.dot(h1, self.W_hr) + gate * cr + self.b_r)
        h_t = K.tanh(r_t * K.dot(h1, self.W_hh) + gate * ch + self.b_h)

        h_t = z_t * h1 + (1. - z_t) * h_t
        h_t = x_m * h_t + (1. - x_m) * h1

        return [h_t]


    # for fertility model
    def _get_fertility(self, c):
        fertility = K.sigmoid(K.dot(c, self.W_cov_fertility) + self.b_cov_fertility) * self.max_fertility
        c_shape = K.shape(c)
        fertility = K.reshape(fertility, shape=(c_shape[0], c_shape[1]))
        return fertility


    def _update_coverage(self, cov_tm1, probs, c, h_tm1, fertility=None):
        '''
        cov_tm1:    coverage at time (t-1)
        probs:      attention probabilities at time t
        c:          source annotations
        fertility:  fertility of individual source word
        '''
        if self.coverage_type is 'linguistic':
            assert fertility, 'ferility should be given for linguistic coverage'
            fertility_probs = probs / fertility
            # TODO: verify why broadcast 2 and then unbroadcast?
            cov = K.expand_dims(fertility_probs)

            # accumulation
            cov = cov_tm1 + cov
        else:
            # we can precompute w*c in advance to minimize the computational cost
            extend_probs = K.expand_dims(probs)
            z = K.sigmoid(K.dot(cov_tm1, self.W_cov_z) + K.dot(extend_probs, self.W_cov_pz) + K.dot(c, self.W_cov_cz) + K.dot(h_tm1, self.W_cov_hz) + self.b_cov_z)
            r = K.sigmoid(K.dot(cov_tm1, self.W_cov_r) + K.dot(extend_probs, self.W_cov_pr) + K.dot(c, self.W_cov_cr) + K.dot(h_tm1, self.W_cov_hr) + self.b_cov_r)
            cov = K.tanh(r * K.dot(cov_tm1, self.W_cov_h) + K.dot(extend_probs, self.W_cov_ph) + K.dot(c, self.W_cov_ch) + K.dot(h_tm1, self.W_cov_hh) + self.b_cov_h)
            cov = (1 - z) * cov_tm1 + z * cov

        return cov


    def _step_attention(self, x_h, x_z, x_r, x_m, h_tm1, c, c_m, p_from_c, cov_tm1=None, fertility=None):
        '''
        x_h: input at time t
        x_z: update of input
        x_r: reset of input
        x_m: mask of x_t
        h_tm1: previous state
        cov_tm1:  coverage at time (t-1)
        fertility:  fertility of individual source word
        '''

        # here h1 combines previous hidden state and lastly generated word with GRU
        # note that this is different from the paper
        z1 = K.sigmoid(K.dot(h_tm1, self.W_n1_z) + x_z + self.b_n1_z)
        r1 = K.sigmoid(K.dot(h_tm1, self.W_n1_r) + x_r + self.b_n1_r)
        h1 = K.tanh(r1 * K.dot(h_tm1, self.W_n1_h) + x_h + self.b_n1_h)
        # nb_samples, n_hids
        h1 = z1 * h_tm1 + (1. - z1) * h1
        h1 = x_m * h1 + (1. - x_m) * h_tm1

        # 1, nb_samples, dim
        p_from_h = K.expand_dims(K.dot(h1, self.B_hp) + self.b_tt, axis=0)
        # time_stpes, nb_samples, dim
        p = p_from_h + p_from_c


        if self.with_coverage:
            p_from_cov = K.dot(cov_tm1, self.C_covp)
            p += p_from_cov

        # energy = exp(dot(tanh(p), self.D_pe) + self.c_tt).reshape((source_len, target_num))
        # since self.c_tt has nothing to do with the probs, why? since it contributes an e^c_tt() to the the denominator and nominator
        # note: self.D_pe has a shape of (hidden_output_dim,1)
        # time_steps, nb_samples, 1
        energy = K.exp(K.dot(K.tanh(p), self.D_pe))

        # c_m: time_steps, nb_samples
        if c_m is not None:
            energy *= c_m

        normalizer = K.sum(energy, axis=0, keepdims=True)
        probs = energy / normalizer
        probs = K.squeeze(probs, axis=2)

        ctx = K.sum(c * K.expand_dims(probs), axis=0)

        # update coverage after producing attention probabilities at time t
        if self.with_coverage:
            cov = self._update_coverage(cov_tm1, probs, c, h_tm1, fertility)


        # this is even more consistent with our context gate
        # h1 corresponds to target context, while ctx corresponds to source context

        if self.with_context_gate:
            gate = K.sigmoid(K.dot(h1, self.W_ctx_h) +
                                  K.dot(ctx, self.W_ctx_c) + self.b_ctx)

            # we directly scale h1, since it used in computing both can_h_t and h_t
            h1 = h1 * (1. - gate)
        else:
            gate = 1.

        z_t = K.sigmoid(K.dot(h1, self.W_hz) + gate * K.dot(ctx, self.W_cz) + self.b_z)
        r_t = K.sigmoid(K.dot(h1, self.W_hr) + gate * K.dot(ctx, self.W_cr) + self.b_r)
        h_t = K.tanh(r_t * K.dot(h1, self.W_hh) + gate * K.dot(ctx, self.W_ch) + self.b_h)

        h_t = z_t * h1 + (1. - z_t) * h_t
        h_t = x_m * h_t + (1. - x_m) * h1

        if self.with_coverage:
            return [h_t, ctx, probs, cov]
        else:
            return [h_t, ctx, probs]


    def create_init_state(self, init_context):
        init_state = K.tanh(K.dot(init_context, self.W_c_init) + self.b_c_init)
        return init_state


    def apply(self, state_below, mask_below=None, init_state=None,
              init_context=None, c=None, c_mask=None, one_step=False,
              cov_before=None, fertility=None):

        # assert c, 'Context must be provided'
        # assert c.ndim == 3, 'Context must be 3-d: n_seq * batch_size * dim'

        # state_below: n_steps * batch_size/1 * embedding

        # mask
        if mask_below is None:    # sampling or beamsearch
            mask_below = K.ones_like(K.sum(state_below, axis=-1, keepdims=True))    # nb_samples

        if K.ndim(mask_below) != K.ndim(state_below):
            mask_below = K.expand_dims(mask_below)

        assert  K.ndim(mask_below) == K.ndim(state_below)

        if one_step:
            assert init_state is not None, 'previous state must be provided'

        if init_state is None:
            init_state = self.create_init_state(init_context)

        state_below_xh = K.dot(state_below, self.W_xh)
        state_below_xz = K.dot(state_below, self.W_xz)
        state_below_xr = K.dot(state_below, self.W_xr)

        if self.with_attention:
            # time steps, nb_samples, n_hids
            p_from_c = K.reshape(K.dot(c, self.A_cp), shape=(K.shape(c)[0], K.shape(c)[1], self.n_hids))
        else:
            c_z = K.dot(init_context, self.W_cz)
            c_r = K.dot(init_context, self.W_cr)
            c_h = K.dot(init_context, self.W_ch)

        if one_step:
            if self.with_attention:
                return self._step_attention(state_below_xh, state_below_xz, state_below_xr,
                                            mask_below, init_state, c, c_mask, p_from_c,
                                            cov_tm1=cov_before, fertility=fertility)
            else:
                return self._step_context(state_below_xh, state_below_xz, state_below_xr,
                                          mask_below, init_state, c_z, c_r, c_h, init_context)
        else:
            sequences = [state_below_xh, state_below_xz, state_below_xr, mask_below]
            # decoder hidden state
            outputs_info = [init_state]
            if self.with_attention:
                # ctx, probs
                if K._BACKEND == 'theano':
                    outputs_info += [None, None]
                else:
                    outputs_info += [K.zeros_like(K.sum(c, axis=0)), K.zeros_like(K.sum(c, axis=-1))]

                if self.with_coverage:
                    # initialization for coverage
                    # TODO: check c is 3D
                    init_cov = K.repeat_elements(K.expand_dims(K.zeros_like(K.sum(c, axis=2))), self.coverage_dim, axis=2)
                    outputs_info.append(init_cov)
                    # fertility is not constructed outside when training
                    if self.coverage_type is 'linguistic':
                        fertility = self._get_fertility(c)
                    else:
                        fertility = K.zeros_like(K.sum(c, axis=2))
                    if K._BACKEND == 'theano':
                        fn = lambda  x_h, x_z, x_r, x_m, h_tm1, cov_tm1 :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_mask, p_from_c, cov_tm1=cov_tm1, fertility=fertility)
                    else:
                        fn = lambda  (h_tm1, ctx_tm1, probs_tm1, cov_tm1), (x_h, x_z, x_r, x_m) :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_mask, p_from_c, cov_tm1=cov_tm1, fertility=fertility)
                else:
                    if K._BACKEND == 'theano':
                        if self.mkl == True:
                            print('with mkl')
                            #alignment GRU
                            W_x_a = K.concatenate([self.W_xh, self.W_xz, self.W_xr], axis=0)
                            W_h_a = K.concatenate([self.W_n1_z, self.W_n1_r, self.W_n1_h], axis=0)
                            b_a = K.concatenate([self.b_n1_z, self.b_n1_r, self.b_n1_h], axis=0)
                            hidden_alignment = self.GRU_op(state_below,W_x_a, W_h_a, init_state, b_a)[0]
                            #attention
                            ctx, probs = self.attention_.apply(hidden_alignment, c, p_from_c)
                            #decoder GRU 
                            W_x_c = K.concatenate([self.W_cz, self.W_cr, self.W_ch], axis=0)
                            W_h_c = K.concatenate([self.W_hz, self.W_hr, self.W_hh], axis=0)
                            b_c = K.concatenate([self.b_z, self.b_r, self.b_h], axis=0)
                            init = hidden_alignment[K.shape(hidden_alignment)[0] - 1,:,:]
                            hidden_decoder = self.GRU_op(ctx, W_x_c, W_h_c, init, b_c)[0]
                            
                            self.output = [hidden_decoder, ctx, probs]    
                        else:
                            fn = lambda  x_h, x_z, x_r, x_m, h_tm1 :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_mask, p_from_c)
                    else:
                        fn = lambda  (h_tm1, ctx_tm1, probs_tm1), (x_h, x_z, x_r, x_m) :  self._step_attention(x_h, x_z, x_r, x_m, h_tm1, c, c_mask, p_from_c)
            
            else:
                if K._BACKEND == 'theano':
                    fn = lambda  x_h, x_z, x_r, x_m, h_tm1 :  self._step_context(x_h, x_z, x_r, x_m, h_tm1, c_z, c_r, c_h, init_context)
                else:
                    fn = lambda  (h_tm1,), (x_h, x_z, x_r, x_m) :  self._step_context(x_h, x_z, x_r, x_m, h_tm1, c_z, c_r, c_h, init_context)

            if self.mkl == False:
                self.output = K.scan(fn,
                                 sequences=sequences,
                                 outputs_initials=outputs_info,
                                 name=_p(self.pname, 'layers'))

            return self.output


    def readout(self, hiddens, ctxs, state_below):

        readout = K.dot(hiddens, self.W_o_h) + \
                  K.dot(ctxs, self.W_o_c) + \
                  K.dot(state_below, self.W_o_e) + \
                  self.b_o

        return K.tanh(readout)


    def one_step_maxout(self, readout):
        readout_shape = K.shape(readout)
        maxout = K.max(K.reshape(readout, shape=(readout_shape[0],
                                  readout_shape[1] / self.maxout_part,
                                  self.maxout_part)), axis=2)

        return maxout


    def run_pipeline(self, state_below, mask_below, init_context=None, c=None, c_mask=None):

        init_state = self.create_init_state(init_context)

        if self.with_attention:
            # [hiddens, ctxs] = self.apply(state_below=state_below, mask_below=mask_below,
            results = self.apply(state_below=state_below,
                                 mask_below=mask_below,
                                 init_state=init_state,
                                 c=c,
                                 c_mask=c_mask)
            hiddens = results[0]
            ctxs = results[1]
            probs = results[2]
        else:
            hiddens = self.apply(state_below=state_below, mask_below=mask_below,
                                 init_context=init_context,
                                 init_state=init_state, c=c, c_mask=c_mask)
            # TODO shape of state_below, nb_samples, time_steps, input_dim?
            n_times = K.shape(state_below)[0]
            ctxs = ReplicateLayer(init_context, n_times)
            probs = None

        # readout
        readout = self.readout(hiddens, ctxs, state_below)

        # maxout
        if self.maxout_part > 1:
            readout_shape = K.shape(readout)
            readout = K.max(K.reshape(readout, shape=(readout_shape[0],
                                                      readout_shape[1],
                                                      readout_shape[2] / self.maxout_part,
                                                      self.maxout_part)),
                            axis=3)


        # for reconstruction, we need decoder states
        # return readout * mask_below[:, :, None]
        return hiddens, readout * mask_below, probs


# for reconstruction
class InverseDecoder(Decoder):
    def __init__(self, n_in, n_hids, n_cdim, maxout_part=2,
                 name='rnn_inverse_decoder', with_attention=True):

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_cdim = n_cdim
        self.maxout_part = maxout_part
        self.pname = name
        self.with_attention = with_attention
        self.with_coverage = False
        self.with_context_gate = False
        self._init_params()


    def _init_params(self):
        # generally, parameters with shape shape_ch = (self.c_ndim, self.n_hids) can be applied with tied weights
        # this for combining lastly generated words and decoder state,
        # and thus cannot be applied with tied weights
        shape_xh = (self.n_in, self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)

        self.W_xz = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xz'))
        self.W_xr = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xr'))
        self.W_xh = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_h'))
        self.W_hz = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hz'))
        self.W_hr = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hr'))
        self.W_hh = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        shape_ch = (self.n_cdim, self.n_hids)

        self.W_cz = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cz'))
        self.W_cr = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cr'))
        self.W_ch = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_ch'))
        self.W_c_init = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_c_init'))
        # we don't add the new params if we use tied_weights, since we reuse the weights in decoder
        self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init]

        self.b_c_init = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_c_init'))
        self.params += [self.b_c_init]


        # we moved the parameters below here, to make it works for both with_context and with_attention modes
        # modification in this version
        # in the paper, e_{i,j} = a(s_{i-1}, h_j)
        # here, e_{i,j} = a(GRU(s_{i-1}, y_{i-1}), h_j), which considers the lastly generated target word
        # all the following parameters are for the introduced GRU
        # it is reasonable
        self.W_n1_h = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_h'))
        self.W_n1_r = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_r'))
        self.W_n1_z = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_n1_z'))
        self.b_n1_h = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_h'))
        self.b_n1_r = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_r'))
        self.b_n1_z = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_n1_z'))
        self.params += [self.W_n1_h, self.W_n1_r, self.W_n1_z, self.b_n1_h, self.b_n1_r, self.b_n1_z]
        ###############################################

        if self.with_attention:
            self.A_cp = norm_weight(shape=shape_ch, name=_p(self.pname, 'A_cp'))
            self.params += [self.A_cp]

            self.B_hp = norm_weight(shape=shape_hh, name=_p(self.pname, 'B_hp'))
            self.b_tt = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_tt'))
            self.D_pe = norm_weight(shape=(self.n_hids, 1), name=_p(self.pname, 'D_pe'))
            # self.c_tt = constant_weight(shape=(1,), name=_p(self.pname, 'c_tt'))
            self.params += [self.B_hp, self.b_tt, self.D_pe]

        # for error on encoder states, we don't need the probability
        # thus no need for readout, which costs a large number of parameters
        # for readout
        n_out = self.n_in * self.maxout_part
        self.W_o_c = norm_weight(shape=(self.n_cdim, n_out), name=_p(self.pname, 'W_out_c'))
        self.W_o_h = norm_weight(shape=(self.n_hids, n_out), name=_p(self.pname, 'W_out_h'))
        self.W_o_e = norm_weight(shape=(self.n_in, n_out), name=_p(self.pname, 'W_out_e'))
        self.b_o = constant_weight(shape=(n_out,), name=_p(self.pname, 'b_out_o'))

        self.params += [self.W_o_c, self.W_o_h, self.W_o_e, self.b_o]


    def run_pipeline(self, state_below, mask_below, init_context=None, c=None, c_mask=None):

        init_state = self.create_init_state(init_context)

        if self.with_attention:

            results = self.apply(state_below=state_below, mask_below=mask_below,
                                 init_state=init_state, c=c, c_mask=c_mask)
            hiddens = results[0]
            ctxs = results[1]
            probs = results[2]
        else:
            hiddens = self.apply(state_below=state_below, mask_below=mask_below,
                                 init_context=init_context,
                                 init_state=init_state, c=c, c_mask=c_mask)
            n_times = K.shape(state_below)[0]
            # TODO: verify
            ctxs = ReplicateLayer(init_context, n_times)
            probs = None

        # readout
        readout = self.readout(hiddens, ctxs, state_below)

        # maxout
        if self.maxout_part > 1:
            readout_shape = K.shape(readout)
            readout = K.max(K.reshape(readout, shape=(readout_shape[0],
                                       readout_shape[1],
                                       readout_shape[2] / self.maxout_part,
                                       self.maxout_part)), axis=3)

        return hiddens, readout * mask_below, probs


class LookupTable(object):

    def __init__(self, vocab_size, embedding_size, name='embeddings'):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # for norm_weight
        self.W = norm_weight(shape=(vocab_size, embedding_size), name=name)

        # parameters of the model
        self.params = [self.W]


    def apply(self, indices):
        return K.gather(self.W, indices)



class LogisticRegression(object):

    """Multi-class Logistic Regression Class"""

    def __init__(self, n_in, n_out, name='LR'):
        self.W = norm_weight(shape=(n_in, n_out), name=_p(name, 'W'))
        self.b = constant_weight(shape=(n_out,), name=_p(name, 'b'))
        self.params = [self.W, self.b]
        self.n_out = n_out

    def get_logits(self, x):
        return  K.dot(x, self.W) + self.b

    def get_logits_with_multiple_devices(self, x, ps_device, devices):
        assert K._BACKEND == 'tensorflow'
        import tensorflow as tf
        num_devices = len(devices)
        with tf.device(ps_device):
            w_splits = tf.split(1, num_devices, self.W)
        y_list = []
        for w_split, device in zip(w_splits, devices):
            with tf.device(device):
                y = K.dot(x, w_split)
                y_list.append(y)

        # merge along the last dimension
        with tf.device(devices[0]):
            logits = K.concatenate(y_list)

        return logits

class GRU(object):

    def __init__(self, n_in, n_hids, name='GRU', with_context=False):

        self.n_in = n_in
        self.n_hids = n_hids
        self.pname = name

        self.with_context = with_context
        if self.with_context:
            self.c_hids = n_hids

        self._init_params()


    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)

        self.W_xz = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xz'))
        self.W_xr = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xr'))
        self.W_xh = norm_weight(shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_z = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_z'))
        self.b_r = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_r'))
        self.b_h = constant_weight(shape=(self.n_hids,), name=_p(self.pname, 'b_h'))
        self.W_hz = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hz'))
        self.W_hr = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hr'))
        self.W_hh = ortho_weight(shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_context:
            shape_ch = (self.c_hids, self.n_hids)
            self.W_cz = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cz'))
            self.W_cr = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_cr'))
            self.W_ch = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_ch'))
            self.W_c_init = norm_weight(shape=shape_ch, name=_p(self.pname, 'W_c_init'))

            self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init]

    def _step(self, x_h, x_z, x_r, x_m, h_tm1):
        '''
        x_h: input at time t
        x_z: update for x_t
        x_r: reset for x_t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        z_t = K.sigmoid(x_z + K.dot(h_tm1, self.W_hz) + self.b_z)
        r_t = K.sigmoid(x_r + K.dot(h_tm1, self.W_hr) + self.b_r)
        can_h_t = K.tanh(x_h + r_t * K.dot(h_tm1, self.W_hh) + self.b_h)

        h_t = (1. - z_t) * h_tm1 + z_t * can_h_t
        h_t = x_m * h_t + (1. - x_m) * h_tm1
        return h_t

    def _step_context(self, x_t, x_m, h_tm1, cz, cr, ch):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        z_t = K.sigmoid(K.dot(x_t, self.W_xz) +
                             K.dot(h_tm1, self.W_hz) +
                             K.dot(cz, self.W_cz) + self.b_z)

        r_t = K.sigmoid(K.dot(x_t, self.W_xr) +
                             K.dot(h_tm1, self.W_hr) +
                             K.dot(cr, self.W_cr) + self.b_r)

        can_h_t = K.tanh(K.dot(x_t, self.W_xh) +
                         r_t * K.dot(h_tm1, self.W_hh) +
                         K.dot(ch, self.W_ch) + self.b_h)

        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t
        h_t = x_m * h_t + (1 - x_m) * h_tm1
        return h_t

    def apply(self, state_below, mask_below=None, init_state=None, context=None):

        if K.ndim(state_below) == 2:
            state_below = K.expand_dims(state_below, 1)

        if mask_below is None:
            mask_below = K.ones_like(K.sum(state_below, axis=2, keepdims=True))

        if init_state is None:
            # nb_samples,n_hids
            init_state = K.repeat_elements(K.expand_dims(K.zeros_like(K.sum(state_below, axis=[0, 2]))), self.n_hids, axis=1)
        print('init state ',K.ndim(init_state))

        state_below_xh = K. dot(state_below, self.W_xh)
        state_below_xz = K. dot(state_below, self.W_xz)
        state_below_xr = K.dot(state_below, self.W_xr)
        sequences = [state_below_xh, state_below_xz, state_below_xr, mask_below]

        if K._BACKEND == 'theano':
            fn = lambda x_h, x_z, x_r, x_m, h_tm1: self._step(x_h, x_z, x_r, x_m, h_tm1)
        else:
            fn = lambda h_tm1, (x_h, x_z, x_r, x_m): self._step(x_h, x_z, x_r, x_m, h_tm1)

        rval = K.scan(fn,
                      sequences=sequences,
                      outputs_initials=init_state,
                      name=_p(self.pname, 'layers'))

        self.output = rval
        return self.output


    def run_pipeline(self, state_below, mask_below, context=None):

        hiddens = self.apply(state_below, mask_below, context=context)

        if self.with_context:
            n_in = self.n_in + self.n_hids + self.c_hids
            n_out = self.n_hids * 2
            n_times = K.shape(state_below)[0]
            r_context = ReplicateLayer(context, n_times)
            combine = K.concatenate([state_below, hiddens, r_context], axis=2)
        else:
            n_in = self.n_in + self.n_hids
            n_out = self.n_hids * 2    # for maxout
            combine = K.concatenate([state_below, hiddens], axis=2)

        self.W_m = norm_weight(shape=(n_in, n_out), name=_p(self.pname, 'W_m'))
        self.b_m = constant_weight(shape=(n_out,), name=_p(self.pname, 'b_m'))

        self.params += [self.W_m, self.b_m]

        # maxout
        merge_out = K.dot(combine, self.W_m) + self.b_m
        merge_out_shape = K.shape(merge_out)
        merge_max_out = K.max(K.reshape(merge_out, shape=(merge_out_shape[0],
                                       merge_out_shape[1],
                                       merge_out_shape[2] / 2,
                                       2)), axis=3)

        return merge_max_out * mask_below

