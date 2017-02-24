# rnn encoder-decoder for machine translation
import numpy
import keras.backend as K
import backend
from tensorflow.contrib.layers.python.layers.regularizers import l1_regularizer

K.dot = backend.dot
K.shift_right = backend.shift_right
K.foreach = backend.foreach
K.random_multinomial = backend.random_multinomial
import logging
import os
from models import LookupTable, LogisticRegression, BidirectionalEncoder, Decoder, InverseDecoder
from utils import Dropout
from algorithm import adadelta, grad_clip

logger = logging.getLogger(__name__)

# TODO: move to backend



if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def avg_grads(grads_list):
        grads = grads_list[0]
        num_grads = len(grads)
        for other_grads in grads_list[1:]:
            for i in xrange(num_grads):
                grads[i] = tf.add(grads[i], other_grads[i])
        for i in xrange(num_grads):
            grads[i] = tf.div(grads[i], num_grads)
        return grads

    tf.set_random_seed(20080524)

    def sampled_softmax_loss(weights,
                             biases,
                             num_sampled,
                             num_classes,
                             labels,
                             inputs,
                             mask=None,
                             num_true=1,
                             sampled_values=None,
                             remove_accidental_hits=True):
        """Computes and returns the sampled softmax training loss.
        This is a faster way to train a softmax classifier over a huge number of
        classes.
        This operation is for training only.  It is generally an underestimate of
        the full softmax loss.
        At inference time, you can compute full softmax probabilities with the
        expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.
        See our [Candidate Sampling Algorithms Reference]
        (../../extras/candidate_sampling.pdf)
        Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
        ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.
        Args:
          weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
              objects whose concatenation along dimension 0 has shape
              [num_classes, dim].  The (possibly-sharded) class embeddings.
          biases: A `Tensor` of shape `[num_classes]`.  The class biases.
          inputs: A `Tensor` of shape `[time steps, batch_size, dim]`.  The forward
              activations of the input network.
          mask: A tensor of shape [time_steps, batch_size,1].
          labels: A `Tensor` of type `int64` and shape `[time_steps,batch_size,
              num_true]`. The target classes.  Note that this format differs from
              the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
          num_sampled: An `int`.  The number of classes to randomly sample per batch.
          num_classes: An `int`. The number of possible classes.
          num_true: An `int`.  The number of target classes per training example.
          sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
              `sampled_expected_count`) returned by a `*_candidate_sampler` function.
              (if None, we default to `log_uniform_candidate_sampler`)
          remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
              where a sampled class equals one of the target classes.  Default is
              True.
          partition_strategy: A string specifying the partitioning strategy, relevant
              if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
              Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
          name: A name for the operation (optional).
        Returns:
          A `batch_size` 1-D tensor of per-example sampled softmax losses.
        """
        assert K.ndim(inputs) == 3    # time_steps, number_samples, input_dim
        nb_samples = K.cast(K.shape(inputs)[1], K.dtype(weights))

        inputs = K.reshape(inputs, (-1, K.shape(inputs)[2]))
        labels = K.reshape(labels, (-1, 1))
        labels = K.cast(labels, 'int64')

        ce = tf.nn.sampled_softmax_loss(weights=weights,
                                          biases=biases,
                                          inputs=inputs,
                                          labels=labels,
                                          num_sampled=num_sampled,
                                          num_classes=num_classes,
                                          num_true=num_true,
                                          sampled_values=sampled_values,
                                          remove_accidental_hits=remove_accidental_hits)
        if mask is not None:
            mask_flat = K.flatten(mask)    # time_steps*nb_samples
            ce *= mask_flat

        return K.sum(ce) / nb_samples


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

def get_category_cross_entropy_from_flat_logits(logits_flat, targets, mask=None):
    assert K.ndim(targets) == 2    # time_steps * nb_samples
    nb_samples = K.cast(K.shape(targets)[1], K.dtype(logits_flat))

    targets_flat = K.flatten(targets)
    if K._BACKEND == 'tensorflow':
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, K.cast(targets_flat, 'int64'))
    else:
        # Theano will internally call one hot version if the two dims do not match
        ce = K.categorical_crossentropy(logits_flat, targets_flat, from_logits=True)
    if mask is not None:
        mask_flat = K.flatten(mask)
        ce *= mask_flat

    return K.sum(ce) / nb_samples

def get_probs_from_logits(logits):
    logits_shape = K.shape(logits)
    logits_flat = K.reshape(logits, shape=(-1, logits_shape[K.ndim(logits) - 1]))
    probs_flat = K.softmax(logits_flat)
    return K.reshape(probs_flat, shape=logits_shape)

# TODO: apply sampled_softmax to re-construction loss

def calc_loss_from_readout(readout, targets, targets_mask, logisticRegressionLayer, softmax_output_num_sampled=100000):
    n_out = logisticRegressionLayer.n_out
    if n_out >= softmax_output_num_sampled and K._BACKEND == 'tensorflow':
        logger.info('Used sampled_softmax with number of class samples = {}'.format(softmax_output_num_sampled))
        cost = sampled_softmax_loss(weights=K.transpose(logisticRegressionLayer.W),
                             biases=logisticRegressionLayer.b,
                             num_sampled=softmax_output_num_sampled,
                             num_classes=n_out,
                             labels=targets,
                             inputs=readout,
                             mask=targets_mask)
    else:
        logits = logisticRegressionLayer.get_logits(readout)
        logits_flat = K.reshape(logits, shape=(-1, n_out))
        cost = get_category_cross_entropy_from_flat_logits(logits_flat, targets, targets_mask)
    return cost

class EncoderDecoder(object):

    def __init__(self, **kwargs):
        self.n_in_src = kwargs.pop('nembed_src')
        self.n_in_trg = kwargs.pop('nembed_trg')
        self.n_hids_src = kwargs.pop('nhids_src')
        self.n_hids_trg = kwargs.pop('nhids_trg')
        self.src_vocab_size = kwargs.pop('src_vocab_size')
        self.trg_vocab_size = kwargs.pop('trg_vocab_size')
        self.method = kwargs.pop('method')
        self.dropout = kwargs.pop('dropout')
        self.maxout_part = kwargs.pop('maxout_part')
        self.path = kwargs.pop('saveto')
        self.clip_c = kwargs.pop('clip_c')

        self.with_attention = kwargs.pop('with_attention')

        self.with_coverage = kwargs.pop('with_coverage')
        self.coverage_dim = kwargs.pop('coverage_dim')
        self.coverage_type = kwargs.pop('coverage_type')
        self.max_fertility = kwargs.pop('max_fertility')
        if self.coverage_type is 'linguistic':
            # make sure the dimension of linguistic coverage is always 1
            self.coverage_dim = 1

        self.with_context_gate = kwargs.pop('with_context_gate')

        self.params = []
        self.layers = []

        self.table_src = LookupTable(self.src_vocab_size, self.n_in_src, name='table_src')
        self.layers.append(self.table_src)

        self.encoder = BidirectionalEncoder(self.n_in_src, self.n_hids_src, self.table_src, name='birnn_encoder')
        self.layers.append(self.encoder)

        self.table_trg = LookupTable(self.trg_vocab_size, self.n_in_trg, name='table_trg')
        self.layers.append(self.table_trg)

        self.decoder = Decoder(self.n_in_trg,
                               self.n_hids_trg,
                               2 * self.n_hids_src,
                               with_attention=self.with_attention,
                               with_coverage=self.with_coverage,
                               coverage_dim=self.coverage_dim,
                               coverage_type=self.coverage_type,
                               max_fertility=self.max_fertility,
                               with_context_gate=self.with_context_gate,
                               maxout_part=self.maxout_part,
                               name='rnn_decoder')

        self.layers.append(self.decoder)
        self.logistic_layer = LogisticRegression(self.n_in_trg, self.trg_vocab_size)
        self.layers.append(self.logistic_layer)

        # for reconstruction
        self.with_reconstruction = kwargs.pop('with_reconstruction')
        self.with_tied_weights = kwargs.pop('with_tied_weights')

        self.reconstruction_weight = kwargs.pop('reconstruction_weight')

        # we encourage the agreement for bidirectional attention, inspired by Cheng et al., (2016)
        self.with_attention_agreement = kwargs.pop('with_attention_agreement')
        self.attention_agreement_weight = kwargs.pop('attention_agreement_weight')

        self.with_reconstruction_error_on_states = kwargs.pop('with_reconstruction_error_on_states')
        if self.with_reconstruction:
            self.with_reconstruction_coverage = kwargs.pop('with_reconstruction_coverage')
            self.with_reconstruction_context_gate = kwargs.pop('with_reconstruction_context_gate')
            # note the source and target sides are reversed
            self.inverse_decoder = InverseDecoder(self.n_in_src, 2 * self.n_hids_src, self.n_hids_trg,
                                   with_attention=self.with_attention,
                                   with_coverage=self.with_reconstruction_coverage,
                                   coverage_dim=self.coverage_dim,
                                   coverage_type=self.coverage_type,
                                   max_fertility=self.max_fertility,
                                   with_context_gate=self.with_reconstruction_context_gate,
                                   maxout_part=self.maxout_part, name='rnn_inverse_decoder',
                                   with_tied_weights=self.with_tied_weights,
                                   with_reconstruction_error_on_states=self.with_reconstruction_error_on_states,
                                   decoder=self.decoder)
            self.layers.append(self.inverse_decoder)
            if not self.with_reconstruction_error_on_states:
                self.inverse_logistic_layer = LogisticRegression(self.n_in_src, self.src_vocab_size, name='inverse_LR')
                self.layers.append(self.inverse_logistic_layer)

            # fast training for the new parameters introduced by reconstruction
            self.with_fast_training = kwargs.pop('with_fast_training')
            if self.with_fast_training:
                self.fix_base_parameters = kwargs.pop('fix_base_parameters')
                self.fast_training_iterations = numpy.float32(kwargs.pop('fast_training_iterations'))

                self.new_params = []
                self.new_params.extend(self.inverse_decoder.params)
                if not self.with_reconstruction_error_on_states:
                    self.new_params.extend(self.inverse_logistic_layer.params)
                self.fast_training_parameters = [p.name for p in self.new_params]


        for layer in self.layers:
            self.params.extend(layer.params)

    def build_trainer_with_data_parallel(self, src, src_mask, trg, trg_mask, ite, devices,
                                         l1_reg_weight=1e-6,
                                         l2_reg_weight=1e-6,
                                         softmax_output_num_sampled=100000):

        assert K._BACKEND == 'tensorflow'

        src_mask_3d = K.expand_dims(src_mask)
        trg_mask_3d = K.expand_dims(trg_mask)

        num_devices = len(devices)

        data = [src, src_mask_3d, trg, trg_mask_3d]
        # split along nb_samples dimension
        splitted_data = [tf.split(1, num_devices, item) for item in data ]
        loss_list = []
        fast_train_loss_list = []
        grads_list = []
        fast_grads_list = []

        for i, device in enumerate(devices):
            with tf.device(device):
                loss, fast_train_loss = self.calc_loss(splitted_data[0][i],
                                      splitted_data[1][i],
                                      splitted_data[2][i],
                                      splitted_data[3][i],
                                      l1_reg_weight=l1_reg_weight,
                                      l2_reg_weight=l2_reg_weight,
                                      softmax_output_num_sampled=softmax_output_num_sampled)

                loss_list.append(loss)
                fast_train_loss_list.append(fast_train_loss)
                grads = K.gradients(loss, self.params)
                grads_list.append(grads)
                if self.with_reconstruction and self.with_fast_training:
                    if self.fix_base_parameters and not self.with_tied_weights:
                        fast_grads = K.gradients(fast_train_loss, self.new_params)
                        fast_grads_list.append(fast_grads)

        avg_loss = sum(loss_list) / num_devices
        # use customized version of gradient to enable colocate_gradients with_ops
        # to ensure the gradient are computed by the same device that do the forward computation
        grads = avg_grads(grads_list)
        grads = grad_clip(grads, self.clip_c)
        updates = adadelta(self.params, grads)
        inps = [src, src_mask, trg, trg_mask]

        self.train_fn = K.function(inps,
                                   [avg_loss] + loss_list,
                                   updates=updates,
                                   name='train_function')

        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                avg_fast_train_loss = sum(fast_train_loss_list) / num_devices
                fast_grads = avg_grads(fast_grads_list)
                fast_grads = grad_clip(fast_grads, self.clip_c)
                fast_updates = adadelta(self.new_params, fast_grads)
                self.fast_train_fn = K.function(inps,
                                                [avg_fast_train_loss] + fast_train_loss_list,
                                                updates=fast_updates,
                                                name='fast_train_function')
            else:
                base_training_ratio = K.minimum(ite / self.fast_training_iterations, numpy.float32(1.))
                fast_updates = adadelta(self.params,
                                        grads,
                                        with_fast_training=self.with_fast_training,
                                        fast_training_parameters=self.fast_training_parameters,
                                        base_training_ratio=base_training_ratio)
                fast_inps = inps + [ite]
                self.fast_train_fn = K.function(fast_inps,
                                                [avg_loss] + loss_list,
                                                updates=fast_updates,
                                                name='fast_train_function')

    def calc_loss(self, src, src_mask_3d, trg, trg_mask_3d,
                  l1_reg_weight=1e-6,
                  l2_reg_weight=1e-6,
                  softmax_output_num_sampled=100000):

        annotations = self.encoder.apply(src, src_mask_3d)
        # init_context = annotations[0, :, -self.n_hids_src:]
        # modification #1
        # mean pooling
        init_context = K.sum (annotations * src_mask_3d, axis=0) / K.sum(src_mask_3d, axis=0)

        trg_emb = self.table_trg.apply(trg)
        # shift_right assumes a 3D tensor, and time steps is dimension one
        trg_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(trg_emb, [1, 0, 2])),
                                               [1, 0, 2])

        hiddens, readout, alignment = self.decoder.run_pipeline(state_below=trg_emb_shifted,
                                            mask_below=trg_mask_3d,
                                            init_context=init_context,
                                            c=annotations,
                                            c_mask=src_mask_3d)

        # apply dropout
        if self.dropout > 0.:
            logger.info('Apply dropout with p = {}'.format(self.dropout))
            readout = Dropout(readout, self.dropout)

        cost = calc_loss_from_readout(readout=readout,
                               targets=trg,
                               targets_mask=trg_mask_3d,
                               logisticRegressionLayer=self.logistic_layer,
                               softmax_output_num_sampled=softmax_output_num_sampled)

        if self.with_reconstruction:
            inverse_init_context = K.sum(hiddens * trg_mask_3d, axis=0) / K.sum(trg_mask_3d, axis=0)
            src_emb = self.table_src.apply(src)
            src_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(src_emb, [1, 0, 2])),
                                               [1, 0, 2])

            inverse_hiddens, inverse_readout, inverse_alignment = self.inverse_decoder.run_pipeline(state_below=src_emb_shifted,
                                                mask_below=src_mask_3d,
                                                init_context=inverse_init_context,
                                                c=hiddens,
                                                c_mask=trg_mask_3d)

            if self.with_reconstruction_error_on_states:
                euclidean_distance = K.sum(K.square((inverse_hiddens - annotations) * src_mask_3d), axis=2)
                src_shape = K.shape(src)
                self.reconstruction_cost = K.sum(K.sqrt(euclidean_distance + 1e-6)) / K.cast(src_shape[1], K.dtype(euclidean_distance))
            else:
                if self.dropout > 0.:
                    inverse_readout = Dropout(inverse_readout, self.dropout)

                inverse_logits = self.inverse_logistic_layer.get_logits(inverse_readout)
                inverse_logits_flat = K.reshape(inverse_logits, shape=(-1, self.inverse_logistic_layer.n_out))
                reconstruction_cost = get_category_cross_entropy_from_flat_logits(inverse_logits_flat, src, src_mask_3d)

            cost += reconstruction_cost * self.reconstruction_weight

            if self.with_attention_agreement:
                # Multiplication (MUL): the element-wise multiplication of corresponding matrix cells
                mul = -K.log(K.sum(alignment * K.permute_dimensions(inverse_alignment, (1, 0, 2)), axis=2) + 1e-6)
                attention_agreement_cost = K.sum(mul) / K.cast(K.shape(src)[1], dtype=K.dtype(mul))
                # self.attention_agreement_cost = theano.printing.Print('alignment cost:')(self.attention_agreement_cost)
                cost += attention_agreement_cost * self.attention_agreement_weight

        L1 = sum([K.sum(K.abs(param)) for param in self.params])
        L2 = sum([K.sum(K.square(param)) for param in self.params])

        params_regular = L1 * l1_reg_weight + L2 * l2_reg_weight

        cost += params_regular

        fast_train_cost = None
        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                # for this specific case, we only need to tune re-constructor
                new_L1 = sum([K.sum(K.abs(param)) for param in self.new_params])
                new_L2 = sum([K.sum(K.square(param)) for param in self.new_params])

                new_params_regular = new_L1 * l1_reg_weight + new_L2 * l2_reg_weight

                fast_train_cost = reconstruction_cost * self.reconstruction_weight + new_params_regular
                if self.with_attention_agreement:
                    fast_train_cost += attention_agreement_cost * self.attention_agreement_weight

        return cost, fast_train_cost


    def build_trainer_with_model_parallel(self, src, src_mask, trg, trg_mask, ite, ps_device, devices, l1_reg_weight=1e-6, l2_reg_weight=1e-6):
        assert K._BACKEND == 'tensorflow'

        src_mask_3d = K.expand_dims(src_mask)
        trg_mask_3d = K.expand_dims(trg_mask)

        # compute loss and grads
        loss, fast_train_loss = self.calc_loss_with_model_parallel(src,
                                                                   src_mask_3d,
                                                                   trg,
                                                                   trg_mask_3d,
                                                                   ps_device=ps_device,
                                                                   devices=devices,
                                                                   l1_reg_weight=l1_reg_weight,
                                                                   l2_reg_weight=l2_reg_weight)
        grads = tf.gradients(loss, self.params, colocate_gradients_with_ops=True)
        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                fast_grads = tf.gradients(fast_train_loss, self.new_params, colocate_gradients_with_ops=True)

        grads = grad_clip(grads, self.clip_c)
        updates = adadelta(self.params, grads)
        inps = [src, src_mask, trg, trg_mask]

        self.train_fn = K.function(inps,
                                   [loss],
                                   updates=updates,
                                   name='train_function')

        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                fast_grads = grad_clip(fast_grads, self.clip_c)
                fast_updates = adadelta(self.new_params, fast_grads)
                self.fast_train_fn = K.function(inps,
                                                [fast_train_loss],
                                                updates=fast_updates,
                                                name='fast_train_function')
            else:
                base_training_ratio = K.minimum(ite / self.fast_training_iterations, numpy.float32(1.))
                fast_updates = adadelta(self.params,
                                        grads,
                                        with_fast_training=self.with_fast_training,
                                        fast_training_parameters=self.fast_training_parameters,
                                        base_training_ratio=base_training_ratio)
                fast_inps = inps + [ite]
                self.fast_train_fn = K.function(fast_inps,
                                                [loss],
                                                updates=fast_updates,
                                                name='fast_train_function')

    def calc_loss_with_model_parallel(self, src, src_mask_3d, trg, trg_mask_3d, ps_device, devices, l1_reg_weight=1e-6, l2_reg_weight=1e-6):
        assert K._BACKEND == 'tensorflow'


        with tf.device(devices[0]):

            annotations = self.encoder.apply(src, src_mask_3d)

            init_context = K.sum (annotations * src_mask_3d, axis=0) / K.sum(src_mask_3d, axis=0)

            trg_emb = self.table_trg.apply(trg)
            # shift_right assumes a 3D tensor, and time steps is dimension one
            trg_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(trg_emb, [1, 0, 2])),
                                                   [1, 0, 2])
            hiddens, readout, alignment = self.decoder.run_pipeline(
                                                state_below=trg_emb_shifted,
                                                mask_below=trg_mask_3d,
                                                init_context=init_context,
                                                c=annotations,
                                                c_mask=src_mask_3d)

            if self.dropout > 0.:
                logger.info('Apply dropout with p = {}'.format(self.dropout))
                readout = Dropout(readout, self.dropout)

        logits = self.logistic_layer.get_logits_with_multiple_devices(readout, ps_device, devices)

        with tf.device(devices[0]):
            logits_flat = K.reshape(logits, shape=(-1, self.logistic_layer.n_out))
            cost = get_category_cross_entropy_from_flat_logits(logits_flat, trg, trg_mask_3d)

        if self.with_reconstruction:
            with tf.device(devices[0]):
                inverse_init_context = K.sum(hiddens * trg_mask_3d, axis=0) / K.sum(trg_mask_3d, axis=0)
                src_emb = self.table_src.apply(src)
                src_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(
                                                    src_emb, [1, 0, 2])), [1, 0, 2])
                inverse_hiddens, inverse_readout, inverse_alignment = self.inverse_decoder.run_pipeline(state_below=src_emb_shifted,
                                                    mask_below=src_mask_3d,
                                                    init_context=inverse_init_context,
                                                    c=hiddens,
                                                    c_mask=trg_mask_3d)
            if self.with_reconstruction_error_on_states:
                with tf.device(devices[0]):
                    euclidean_distance = K.sum(K.square((inverse_hiddens - annotations) * src_mask_3d), axis=2)
                    src_shape = K.shape(src)
                    self.reconstruction_cost = K.sum(K.sqrt(euclidean_distance + 1e-6)) / K.cast(src_shape[1], K.dtype(euclidean_distance))
            else:
                with tf.device(devices[0]):
                    if self.dropout > 0.:
                        inverse_readout = Dropout(inverse_readout, self.dropout)

                inverse_logits = self.inverse_logistic_layer.get_logits_with_multiple_devices(inverse_readout, ps_device, devices)
                with tf.device(devices[0]):
                    inverse_logits_flat = K.reshape(inverse_logits, shape=(-1, self.inverse_logistic_layer.n_out))
                    reconstruction_cost = get_category_cross_entropy_from_flat_logits(inverse_logits_flat, src, src_mask_3d)

            with tf.device(devices[0]):
                cost += reconstruction_cost * self.reconstruction_weight

            if self.with_attention_agreement:
                with tf.device(devices[0]):
                    mul = -K.log(K.sum(alignment * K.permute_dimensions(inverse_alignment, (1, 0, 2)), axis=2) + 1e-6)
                    attention_agreement_cost = K.sum(mul) / K.cast(K.shape(src)[1], dtype=K.dtype(mul))
                    cost += attention_agreement_cost * self.attention_agreement_weight

        L1 = sum([K.sum(K.abs(param)) for param in self.params])
        L2 = sum([K.sum(K.square(param)) for param in self.params])

        params_regular = L1 * l1_reg_weight + L2 * l2_reg_weight

        cost += params_regular

        fast_train_cost = None
        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                # for this specific case, we only need to tune re-constructor
                new_L1 = sum([K.sum(K.abs(param)) for param in self.new_params])
                new_L2 = sum([K.sum(K.square(param)) for param in self.new_params])

                new_params_regular = new_L1 * l1_reg_weight + new_L2 * l2_reg_weight

                fast_train_cost = reconstruction_cost * self.reconstruction_weight + new_params_regular
                if self.with_attention_agreement:
                    fast_train_cost += attention_agreement_cost * self.attention_agreement_weight

        return cost, fast_train_cost


    def build_trainer(self, src, src_mask, trg, trg_mask, ite,
                      l1_reg_weight=1e-6,
                      l2_reg_weight=1e-6,
                      softmax_output_num_sampled=100000):

        src_mask_3d = K.expand_dims(src_mask)
        trg_mask_3d = K.expand_dims(trg_mask)

        annotations = self.encoder.apply(src, src_mask_3d)
        # init_context = annotations[0, :, -self.n_hids_src:]
        # modification #1
        # mean pooling
        init_context = K.sum (annotations * src_mask_3d, axis=0) / K.sum(src_mask_3d, axis=0)

        trg_emb = self.table_trg.apply(trg)
        # shift_right assumes a 3D tensor, and time steps is dimension one
        trg_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(trg_emb, [1, 0, 2])),
                                               [1, 0, 2])

        hiddens, readout, alignment = self.decoder.run_pipeline(state_below=trg_emb_shifted,
                                            mask_below=trg_mask_3d,
                                            init_context=init_context,
                                            c=annotations,
                                            c_mask=src_mask_3d)
        # apply dropout
        if self.dropout > 0.:
            logger.info('Apply dropout with p = {}'.format(self.dropout))
            readout = Dropout(readout, self.dropout)

        self.cost = calc_loss_from_readout(readout=readout,
                               targets=trg,
                               targets_mask=trg_mask_3d,
                               logisticRegressionLayer=self.logistic_layer,
                               softmax_output_num_sampled=softmax_output_num_sampled)
        # for reconstruction
        if self.with_reconstruction:
            # now hiddens is the annotations
            inverse_init_context = K.sum(hiddens * trg_mask_3d, axis=0) / K.sum(trg_mask_3d, axis=0)

            src_emb = self.table_src.apply(src)
            src_emb_shifted = K.permute_dimensions(K.shift_right(K.permute_dimensions(src_emb, [1, 0, 2])),
                                               [1, 0, 2])

            inverse_hiddens, inverse_readout, inverse_alignment = self.inverse_decoder.run_pipeline(state_below=src_emb_shifted,
                                                mask_below=src_mask_3d,
                                                init_context=inverse_init_context,
                                                c=hiddens,
                                                c_mask=trg_mask_3d)

            if self.with_reconstruction_error_on_states:
                euclidean_distance = K.sum(K.square((inverse_hiddens - annotations) * src_mask_3d), axis=2)
                # euclidean_distance = theano.printing.Print('euclidean distance:')(euclidean_distance)
                # 1e-6 is eps to avoid NaN problem when using Euclidean distance
                # The gradient of sqrt is undefined (nan) at 0. Use max(x,EPs) to make sure it is a positive number
                # see http://stackoverflow.com/a/31923358/2859669 for more details
                src_shape = K.shape(src)
                self.reconstruction_cost = K.sum(K.sqrt(euclidean_distance + 1e-6)) / K.cast(src_shape[1], K.dtype(euclidean_distance))
            else:
                if self.dropout > 0.:
                    inverse_readout = Dropout(inverse_readout, self.dropout)

                inverse_logits = self.inverse_logistic_layer.get_logits(inverse_readout)
                inverse_logits_flat = K.reshape(inverse_logits, shape=(-1, self.inverse_logistic_layer.n_out))
                self.reconstruction_cost = get_category_cross_entropy_from_flat_logits(inverse_logits_flat, src, src_mask_3d)


            self.cost += self.reconstruction_cost * self.reconstruction_weight


            if self.with_attention_agreement:
                # Multiplication (MUL): the element-wise multiplication of corresponding matrix cells
                mul = -K.log(K.sum(alignment * K.permute_dimensions(inverse_alignment, (1, 0, 2)), axis=2) + 1e-6)
                self.attention_agreement_cost = K.sum(mul) / K.cast(K.shape(src)[1], dtype=K.dtype(mul))
                self.cost += self.attention_agreement_cost * self.attention_agreement_weight



        self.L1 = sum([K.sum(K.abs(param)) for param in self.params])
        self.L2 = sum([K.sum(K.square(param)) for param in self.params])

        params_regular = self.L1 * l1_reg_weight + self.L2 * l2_reg_weight

        # train cost
        train_cost = self.cost + params_regular

        # gradients
        grads = K.gradients(train_cost, self.params)

        # apply gradient clipping here
        grads = grad_clip(grads, self.clip_c)

        # updates
        updates = adadelta(self.params, grads)

        # train function
        inps = [src, src_mask, trg, trg_mask]

        self.train_fn = K.function(inps, [train_cost], updates=updates, name='train_function')


        # for fast training
        if self.with_reconstruction and self.with_fast_training:
            if self.fix_base_parameters and not self.with_tied_weights:
                # for this specific case, we only need to tune re constructor

                self.new_L1 = sum([K.sum(K.abs(param)) for param in self.new_params])
                self.new_L2 = sum([K.sum(K.square(param)) for param in self.new_params])

                new_params_regular = self.new_L1 * l1_reg_weight + self.new_L2 * l2_reg_weight

                # train cost
                fast_train_cost = self.reconstruction_cost * self.reconstruction_weight + new_params_regular
                if self.with_attention_agreement:
                    fast_train_cost += self.attention_agreement_cost * self.attention_agreement_weight

                # gradients
                fast_grads = K.gradients(fast_train_cost, self.new_params)

                # apply gradient clipping here
                fast_grads = grad_clip(fast_grads, self.clip_c)

                # updates
                fast_updates = adadelta(self.new_params, fast_grads)

                self.fast_train_fn = K.function(inps, [fast_train_cost], updates=fast_updates, name='fast_train_function')
            else:
                # for other strategies, we only need to limit the learning ratio of parameters from pre-trained model
                base_training_ratio = K.minimum(ite / self.fast_training_iterations, numpy.float32(1.))
                fast_updates = adadelta(self.params, grads, with_fast_training=self.with_fast_training, fast_training_parameters=self.fast_training_parameters, base_training_ratio=base_training_ratio)
                fast_inps = [src, src_mask, trg, trg_mask, ite]
                self.fast_train_fn = K.function(fast_inps, [train_cost], updates=fast_updates, name='fast_train_function')


    def build_sampler(self):

        # time steps, nb_samples
        x = K.placeholder((None, None), dtype='int32')

        # Build Networks
        # src_mask is None
        c = self.encoder.apply(x, None)    # None,None,None

        init_context = K.mean(c, axis=0)    # None,None

        init_state = self.decoder.create_init_state(init_context)

        outs = [init_state, c]
        if not self.with_attention:
            outs.append(init_context)

        # compile function
        logger.info('Building compile_init_state_and_context function ...')
        self.compile_init_and_context = K.function([x], outs,
                                                        name='compile_init_and_context')
        logger.info('Done')

        if self.with_attention:
            c = K.placeholder((None, None, None))    # None,None,None
            init_context = K.mean(c, axis=0)    # None,None
        else:
            init_context = K.placeholder((None, None))
        # nb_samples
        y = K.placeholder((None,), dtype='int32')
        # nb_samples, state_dim
        cur_state = K.placeholder((None, None))
        # if it is the first word, emb should be all zero, and it is indicated by -1
        trg_emb = lookup_table(self.table_trg.W, y, name='trg_emb')

        # for with_attention=False
        if self.with_attention and self.with_coverage:
            cov_before = K.placeholder(shape=(None, None, None))
            if self.coverage_type is 'linguistic':
                logger.info('Building compile_fertility ...')
                fertility = self.decoder._get_fertility(c)
                self.compile_fertility = K.function([c], [fertility], name='compile_fertility')
                logger.info('Done')
            else:
                fertility = None
        else:
            cov_before = None
            fertility = None

        # apply one step
        # [next_state, ctxs] = self.decoder.apply(state_below=trg_emb,
        results = self.decoder.apply(state_below=trg_emb,
                                     init_state=cur_state,
                                     init_context=None if self.with_attention else init_context,
                                     c=c if self.with_attention else None,
                                     one_step=True,
                                     cov_before=cov_before,
                                     fertility=fertility)
        next_state = results[0]
        if self.with_attention:
            ctxs, alignment = results[1], results[2]
            if self.with_coverage:
                cov = results[3]
        else:
            # if with_attention=False, we always use init_context as the source representation
            ctxs = init_context

        readout = self.decoder.readout(next_state, ctxs, trg_emb)

        # maxout
        if self.maxout_part > 1:
            readout = self.decoder.one_step_maxout(readout)

        # compute the softmax probability
        next_probs = get_probs_from_logits(self.logistic_layer.get_logits(readout))

        # sample from softmax distribution to get the sample
        # TODO: batch_size* nb_classes
        next_sample = K.argmax(K.random_multinomial(pvals=next_probs))

        # compile function
        logger.info('Building compile_next_state_and_probs function ...')
        inps = [y, cur_state]
        if self.with_attention:
            inps.append(c)
        else:
            inps.append(init_context)
        outs = [next_probs, next_state, next_sample]
        if self.with_attention:
            outs.append(alignment)
            if self.with_coverage:
                inps.append(cov_before)
                outs.append(cov)

        self.compile_next_state_and_probs = K.function(inps, outs, name='compile_next_state_and_probs')
        logger.info('Done')

        # for reconstruction
        if self.with_reconstruction:
            # Build Networks
            # trg_mask is None
            if self.with_attention:
                inverse_c = K.placeholder((None, None, None))
                # mean pooling
                inverse_init_context = K.mean(inverse_c, axis=0)
            else:
                inverse_init_context = K.placeholder((None, None))

            inverse_init_state = self.inverse_decoder.create_init_state(inverse_init_context)

            outs = [inverse_init_state]
            if not self.with_attention:
                outs.append(inverse_init_context)

            # compile function
            logger.info('Building compile_inverse_init_state_and_context function ...')
            self.compile_inverse_init_and_context = K.function([inverse_c], outs, name='compile_inverse_init_and_context')
            logger.info('Done')

            # nb_samples
            src = K.placeholder(shape=(None,), dtype='int32')
            # nb_samples, state_dim
            inverse_cur_state = K.placeholder(shape=(None, None))
            # time_steps, nb_samples
            trg_mask = K.placeholder(shape=(None, None))
            # if it is the first word, emb should be all zero, and it is indicated by -1
            src_emb = lookup_table(self.table_src.W, src, name='src_emb')

            # for with_attention=False
            if self.with_attention and self.with_reconstruction_coverage:
                inverse_cov_before = K.placeholder(shape=(None, None, None))
                if self.coverage_type is 'linguistic':
                    inverse_fertility = self.decoder._get_fertility(inverse_c)
                else:
                    inverse_fertility = None
            else:
                inverse_cov_before = None
                inverse_fertility = None

            # apply one step
            inverse_results = self.inverse_decoder.apply(state_below=src_emb,
                                         init_state=inverse_cur_state,
                                         init_context=None if self.with_attention else inverse_init_context,
                                         c=inverse_c if self.with_attention else None,
                                         c_mask=trg_mask,
                                         one_step=True,
                                         cov_before=inverse_cov_before,
                                         fertility=inverse_fertility)
            inverse_next_state = inverse_results[0]
            if self.with_attention:
                inverse_ctxs, inverse_alignment = inverse_results[1], inverse_results[2]
                if self.with_reconstruction_coverage:
                    inverse_cov = inverse_results[3]
            else:
                # if with_attention=False, we always use init_context as the source representation
                inverse_ctxs = init_context

            if not self.with_reconstruction_error_on_states:
                inverse_readout = self.inverse_decoder.readout(inverse_next_state, inverse_ctxs, src_emb)

                # maxout
                if self.maxout_part > 1:
                    inverse_readout = self.inverse_decoder.one_step_maxout(inverse_readout)

                # apply dropout
                if self.dropout > 0.:
                    inverse_readout = Dropout(inverse_readout, self.dropout)

                # compute the softmax probability
                inverse_next_probs = get_probs_from_logits(self.inverse_logistic_layer.get_logits(inverse_readout))

                # sample from softmax distribution to get the sample
                inverse_next_sample = K.argmax(K.random_multinomial(pvals=inverse_next_probs))
            else:
                inverse_next_probs = K.variable(0.)
                inverse_next_sample = K.variable(0.)

            # compile function
            logger.info('Building compile_inverse_next_state_and_probs function ...')
            inps = [src, trg_mask, inverse_cur_state]
            if self.with_attention:
                inps.append(inverse_c)
            else:
                inps.append(inverse_init_context)
            outs = [inverse_next_probs, inverse_next_state, inverse_next_sample]
            if self.with_attention:
                outs.append(inverse_alignment)
                if self.with_reconstruction_coverage:
                    inps.append(inverse_cov_before)
                    outs.append(inverse_cov)

            self.compile_inverse_next_state_and_probs = K.function(inps, outs, name='compile_inverse_next_state_and_probs')
            logger.info('Done')

    def save(self, path=None):
        if path is None:
            path = self.path
        filenpz = open(path, "w")
        # parameter will have different name under tensorflow and theano
        val = dict([(self.norm_para_name(value.name), K.get_value(value)) for _, value in enumerate(self.params)])
        logger.info("save the model {}".format(path))
        numpy.savez(path, **val)
        filenpz.close()

    def norm_para_name(self, name):
        # LR_W:0
        pos = name.find(':')
        if pos != -1:
            return name[:pos]
        else:
            return name

    def hot_fix_parameter_names(self, params):
        new_model_parameters = {}
        for k in params.keys():
            val = params[k]
            new_name = self.norm_para_name(k)
            new_model_parameters[new_name] = val

        return new_model_parameters

    def load(self, path=None):
        if path is None:
            path = self.path
        if os.path.isfile(path):
            logger.info("load params {}".format(path))
            val = numpy.load(path)
            val = self.hot_fix_parameter_names(val)
            for _, param in enumerate(self.params):
                param_name = self.norm_para_name(param.name)
                logger.info('Loading {} with shape {}'.format(param_name, K.get_value(param).shape))
                if param_name not in val.keys():
                    logger.info('Adding new param {} with shape {}'.format(param_name, K.get_value(param).shape))
                    continue
                if K.get_value(param).shape != val[param_name].shape:
                    logger.info("Error: model param != load param shape {} != {}".format(\
                                        K.get_value(param).shape, val[param_name].shape))
                    raise Exception("loading params shape mismatch")
                else:
                    K.set_value(param, val[param_name])
        else:
            logger.warn("file {} does not exist, ignoring load".format(path))