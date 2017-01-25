import os
import json
from keras.backend import _backend
# Devices, the first one is parameter server, separated by ','
devices = os.getenv('DEVICES', '').split(',')
# Get keras backend
def get_keras_backend():
    _keras_base_dir = os.path.expanduser('~')
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = '/tmp'

    _keras_dir = os.path.join(_keras_base_dir, '.keras')

    _BACKEND = 'theano'
    _config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
    if os.path.exists(_config_path):
        _config = json.load(open(_config_path))

        _backend = _config.get('backend', _BACKEND)
        assert _backend in {'theano', 'tensorflow'}
        return _backend

    if 'KERAS_BACKEND' in os.environ:
        _backend = os.environ['KERAS_BACKEND']
        assert _backend in {'theano', 'tensorflow'}
        return _backend

    return None

KERAS_BACKEND = get_keras_backend()

if KERAS_BACKEND == 'tensorflow':
    import tensorflow as tf
    if devices:
        with tf.device(devices[0]):
            import keras.backend as K
            session = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
            K.set_session(session)
    else:
        import keras.backend as K
else:
    import keras.backend as K

import argparse
import logging
import pprint
import time
from data_stream import DStream, get_devtest_stream
from datasets import build_vocabulary_if_needed
from search import BeamSearch
from sampling import Sampler, BleuValidator
from nmt import EncoderDecoder
import configurations

import numpy as np
np.random.seed(20080524)

def ensure_vocabularies(**kwargs):
    # warning: if the vocabularies already exist, they will not be built.
    # Delete and re-run, the vocabularies will be re-created according to current configuration
    vocab_src = kwargs.pop('vocab_src')
    vocab_trg = kwargs.pop('vocab_trg')
    unk = kwargs.pop('unk_token')
    unk_id = kwargs.pop('unk_id')
    eos = kwargs.pop('eos_token')
    eos_id = kwargs.pop('eos_id')
    src_vocab_size = kwargs.pop('src_vocab_size')
    trg_vocab_size = kwargs.pop('trg_vocab_size')
    train_src = kwargs.pop('train_src')
    train_trg = kwargs.pop('train_trg')

    build_vocabulary_if_needed(files=[train_src],
                                            voc_filepath=vocab_src,
                                            encoding='UTF-8',
                                            eos=eos,
                                            eos_id=eos_id,
                                            unk=unk,
                                            unk_id=unk_id,
                                            max_nb_of_vacabulary=src_vocab_size)

    build_vocabulary_if_needed(files=[train_trg],
                                            voc_filepath=vocab_trg,
                                            encoding='UTF-8',
                                            eos=eos,
                                            eos_id=eos_id,
                                            unk=unk,
                                            unk_id=unk_id,
                                            max_nb_of_vacabulary=trg_vocab_size)
def main(configuration, devices=None):
    #  time_steps*nb_samples
    src = K.placeholder(shape=(None, None), dtype='int32')
    src_mask = K.placeholder(shape=(None, None))
    trg = K.placeholder(shape=(None, None), dtype='int32')
    trg_mask = K.placeholder(shape=(None, None))

    # for fast training of new parameters
    ite = K.placeholder(ndim=0)

    enc_dec = EncoderDecoder(**configuration)

    if devices:
        enc_dec.build_trainer_with_data_parallel(devices, src, src_mask, trg, trg_mask, ite)
    else:
        enc_dec.build_trainer(src, src_mask, trg, trg_mask, ite)

    enc_dec.build_sampler()

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

            if devices:
                for i, device in enumerate(devices):
                    logger.info('epoch %d \t updates %d device %s train cost %.4f use time %.4f'
                        % (epoch, iters, device, tc[i + 1], cur_time - last_time))

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

    ensure_vocabularies(**configuration)

    # update batch size accordingly if multiple devices are used, note that the first device is used as parameter server
    if devices:
        configuration['batch_size'] = (len(devices) - 1) * configuration['batch_size']
        logger.info("Batch size updated to %s" % configuration['batch_size'])
        assert K._BACKEND == 'tensorflow'
        with tf.device(devices[0]):
            main(configuration, devices[1:])
    else:
        main(configuration, devices)
