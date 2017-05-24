import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.manual_variable_initialization(True)
import tensorflow as tf

import argparse
import logging
import pprint
import time
from data_stream import DStream, get_devtest_stream

from search import BeamSearch
from sampling import Sampler, BleuValidator
from nmt import EncoderDecoder
import configurations

import numpy as np

np.random.seed(20080524)


def build_model(configuration):
    model = EncoderDecoder(**configuration)
    return model


def calc_loss(
        model,
        src,
        src_mask,
        trg,
        trg_mask,
        configuration):
    l1_reg_weight = configuration['l1_reg_weight']
    l2_reg_weight = configuration['l2_reg_weight']
    softmax_output_num_sampled = configuration['softmax_output_num_sampled']

    src_mask_3d = K.expand_dims(src_mask)
    trg_mask_3d = K.expand_dims(trg_mask)

    loss = model.calc_loss(src,
                           src_mask_3d,
                           trg,
                           trg_mask_3d,
                           l1_reg_weight=l1_reg_weight,
                           l2_reg_weight=l2_reg_weight,
                           softmax_output_num_sampled=softmax_output_num_sampled)

    return loss


def training(loss, is_chief, clip_c, replicas_to_aggregate):
    # Create the gradient descent optimizer with the given learning rate.
    opt = tf.train.AdadeltaOptimizer()
    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=replicas_to_aggregate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Clip gradient norm
    grads_and_vars = opt.compute_gradients(loss)
    clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_c), var) for grad, var in grads_and_vars]
    train_op = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)
    # Hook which handles initialization and queues. Without this hook,training will not work
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    return train_op, global_step, [sync_replicas_hook]


def build_train_fn(model, is_chief, replicas_to_aggregate):
    src = K.placeholder(shape=(None, None), dtype='int32', name='src')
    src_mask = K.placeholder(shape=(None, None), name='src_mask')
    trg = K.placeholder(shape=(None, None), dtype='int32', name='trg')
    trg_mask = K.placeholder(shape=(None, None), name='trg_mask')

    loss = calc_loss(model, src, src_mask, trg, trg_mask, configuration)

    train_op, global_step, hooks = training(loss,
                                            is_chief=is_chief,
                                            clip_c=model.clip_c,
                                            replicas_to_aggregate=replicas_to_aggregate,
                                            )

    # this will cause loss is calculated each session.run, as a result, all related placeholders should be fed
    # hooks.append(tf.train.NanTensorHook(loss))

    train_fn = K.function(inputs=[src, src_mask, trg, trg_mask], outputs=[train_op, global_step, loss])

    return train_fn, hooks


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--ps_hosts",
                        default="localhost:2224",
                        help="ps hosts separated by ','")

    parser.add_argument("--worker_hosts",
                        default="localhost:2222,localhost:2223",
                        help="worker hosts separated by ','")

    parser.add_argument("--task_index",
                        default=0,
                        type=int,
                        help="Index of task within the job")

    parser.add_argument("--job_name",
                        default="worker",
                        help="worker or ps")

    parser.add_argument("--proto",
                        default="get_config_search_coverage",
                        help="Prototype config to use for config")

    parser.add_argument("--state", help="State to use")

    parser.add_argument("--start", type=int, default=0, help="Iterations to start")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == 'ps':
        server.join()

    configuration = getattr(configurations, args.proto)()

    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    worker_device = "/job:worker/task:%d" % args.task_index

    device = tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster)

    is_chief = (args.task_index == 0)

    with tf.device(device):

        enc_dec = build_model(configuration)

        train_fn, hooks = \
            build_train_fn(model=enc_dec,
                           is_chief=is_chief,
                           replicas_to_aggregate=len(worker_hosts))

        enc_dec.build_sampler()

        sample_search = BeamSearch(enc_dec=enc_dec,
                                   configuration=configuration,
                                   beam_size=1,
                                   maxlen=configuration['seq_len_src'], stochastic=True)

        valid_search = BeamSearch(enc_dec=enc_dec,
                                  configuration=configuration,
                                  beam_size=configuration['beam_size'],
                                  maxlen=3 * configuration['seq_len_src'], stochastic=False)

        sampler = Sampler(sample_search, **configuration)

        bleuvalidator = BleuValidator(valid_search, **configuration)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
    )

    with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            hooks=hooks,
            config=sess_config) as sess:

        K.set_session(sess)

        if configuration['reload'] and is_chief:
            enc_dec.load()

        # train data
        ds = DStream(**configuration)

        # valid data
        vs = get_devtest_stream(data_type='valid', input_file=None, **configuration)

        step = args.start
        best_valid_bleu = -1
        best_epoch = -1
        best_step = -1
        max_epochs = configuration['finish_after']

        while not sess.should_stop():
            for epoch in range(max_epochs):
                for x, x_mask, y, y_mask in ds.get_iterator():

                    last_time = time.time()
                    _, step, loss_v = train_fn([x.T, x_mask.T, y.T, y_mask.T])

                    cur_time = time.time()

                    logger.info('epoch %d \t updates %d train cost %.4f use time %.4f'
                                % (epoch, step, loss_v, cur_time - last_time))

                    if not is_chief:
                        continue

                    if step % configuration['sample_freq'] == 0:
                        sampler.apply(x, y)

                    if step % configuration['save_freq'] == 0:
                        enc_dec.save()

                    if step < configuration['val_burn_in']:
                        continue

                    if (step <= configuration['val_burn_in_fine'] and step % configuration['valid_freq'] == 0) \
                            or (step > configuration['val_burn_in_fine'] and step % configuration[
                                'valid_freq_fine'] == 0):
                        valid_bleu = bleuvalidator.apply(vs, configuration['valid_out'])
                        os.system('mkdir -p results/%d' % step)
                        os.system('mv %s* %s results/%d' % (configuration['valid_out'], configuration['saveto'], step))
                        logger.info('valid_test \t epoch %d \t updates %d valid_bleu %.4f'
                                    % (epoch, step, valid_bleu))
                        if valid_bleu > best_valid_bleu:
                            best_valid_bleu = valid_bleu
                            best_epoch = epoch
                            best_step = step
                            enc_dec.save(path=configuration['saveto_best'])

        logger.info('final result: epoch %d \t updates %d valid_bleu_best %.4f'
                    % (best_epoch, best_step, best_valid_bleu))
