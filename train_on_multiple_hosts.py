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

# example to use
# CUDA_VISIBLE_DEVICES= python worker.py --ps_hosts localhost:2224 --worker_hosts localhost:2222,localhost:2223 --job_name ps > ps.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python worker.py --ps_hosts localhost:2224 --worker_hosts localhost:2222,localhost:2223 --job_name worker > worker_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 python worker.py --ps_hosts localhost:2224 --worker_hosts localhost:2222,localhost:2223 --job_name worker --task_index 1 > worker_1.log 2>&1 &

# python train_on_multiple_hosts.py --state config.py --ps_hosts localhost:2224 --worker_hosts localhost:2222,localhost:2223  > tf_worker_0.log 2>&1 &

# python train_on_multiple_hosts.py --state config.py --ps_hosts localhost:2224 --worker_hosts localhost:2222,localhost:2223 --task_index 1 > tf_worker_1.log 2>&1 &

def main(configuration, is_chief=False):

    l1_reg_weight = configuration['l1_reg_weight']
    l2_reg_weight = configuration['l2_reg_weight']
    #  time_steps*nb_samples
    src = K.placeholder(shape=(None, None), dtype='int32')
    src_mask = K.placeholder(shape=(None, None))
    trg = K.placeholder(shape=(None, None), dtype='int32')
    trg_mask = K.placeholder(shape=(None, None))

    # for fast training of new parameters
    ite = K.placeholder(ndim=0)

    enc_dec = EncoderDecoder(**configuration)

    softmax_output_num_sampled = configuration['softmax_output_num_sampled']

    enc_dec.build_trainer(src,
                              src_mask,
                              trg,
                              trg_mask,
                              ite,
                              l1_reg_weight=l1_reg_weight,
                              l2_reg_weight=l2_reg_weight,
                              softmax_output_num_sampled=softmax_output_num_sampled)

    enc_dec.build_sampler()

    # Chief is responsible for initializing and loading model states

    if is_chief:
        init_op = tf.initialize_all_variables()
        init_fn = K.function(inputs=[], outputs=[init_op])
        init_fn([])

        if configuration['reload']:
            enc_dec.load()

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

    # train function
    train_fn = enc_dec.train_fn

    if configuration['with_reconstruction'] and configuration['with_fast_training']:
        fast_train_fn = enc_dec.fast_train_fn

    # train data
    ds = DStream(**configuration)

    # valid data
    vs = get_devtest_stream(data_type='valid', input_file=None, **configuration)

    iters = args.start
    valid_bleu_best = -1
    epoch_best = -1
    iters_best = -1
    max_epochs = configuration['finish_after']

    # TODO: use global iter and only the chief can save the model
    for epoch in range(max_epochs):
        for x, x_mask, y, y_mask in ds.get_iterator():
            last_time = time.time()
            if configuration['with_reconstruction'] and configuration['with_fast_training'] and iters < configuration['fast_training_iterations']:
                if configuration['fix_base_parameters'] and not configuration['with_tied_weights']:
                    tc = fast_train_fn([x.T, x_mask.T, y.T, y_mask.T])
                else:
                    tc = fast_train_fn([x.T, x_mask.T, y.T, y_mask.T, iters])
            else:
                tc = train_fn([x.T, x_mask.T, y.T, y_mask.T])
            cur_time = time.time()
            iters += 1
            logger.info('epoch %d \t updates %d train cost %.4f use time %.4f'
                        % (epoch, iters, tc[0], cur_time - last_time))

            if iters % configuration['save_freq'] == 0:
                enc_dec.save()

            if iters % configuration['sample_freq'] == 0:
                sampler.apply(x, y)

            if iters < configuration['val_burn_in']:
                continue

            if (iters <= configuration['val_burn_in_fine'] and iters % configuration['valid_freq'] == 0) \
               or (iters > configuration['val_burn_in_fine'] and iters % configuration['valid_freq_fine'] == 0):
                valid_bleu = bleuvalidator.apply(vs, configuration['valid_out'])
                os.system('mkdir -p results/%d' % iters)
                os.system('mv %s* %s results/%d' % (configuration['valid_out'], configuration['saveto'], iters))
                logger.info('valid_test \t epoch %d \t updates %d valid_bleu %.4f'
                        % (epoch, iters, valid_bleu))
                if valid_bleu > valid_bleu_best:
                    valid_bleu_best = valid_bleu
                    epoch_best = epoch
                    iters_best = iters
                    enc_dec.save(path=configuration['saveto_best'])

    logger.info('final result: epoch %d \t updates %d valid_bleu_best %.4f'
            % (epoch_best, iters_best, valid_bleu_best))

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

    parser.add_argument("--proto",
                        default="get_config_search_coverage",
                        help="Prototype config to use for config")
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--start", type=int, default=0, help="Iterations to start")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    configuration = getattr(configurations, args.proto)()

    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    prefer_to_model_parallel = configuration['prefer_to_model_parallel']
    logger.info("Prefer_to_model_parallel %s" % prefer_to_model_parallel)

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    is_chief = args.task_index == 0

    host = "grpc://" + worker_hosts[args.task_index]
    session = tf.Session(host)
    K.set_session(session)

    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % args.task_index,
        cluster=cluster)):

        main(configuration, is_chief=is_chief)




