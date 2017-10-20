import os
import json
from keras.backend import _backend
import numpy
import theano
import mlsl
# Devices, the first one is parameter server, separated by ','
devices = [device for device in os.getenv('DEVICES', '').split(',') if device]
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

def split(data, num_part):
    if num_part == 1:
        return [data]
    parts = [[] for _ in xrange(num_part)]
    for i in xrange(len(data)):
        parts[i % num_part].append(data[i])
    parts = [np.array(i, dtype=data.dtype) for i in parts]
    return parts

def main(configuration, ps_device=None, devices=None):
    mkl_multinode = configuration['mkl_multinode']
    if mkl_multinode == True:
        mlsl_obj = mlsl.MLSL()
        mlsl_obj.init()
        node_idx = mlsl_obj.get_process_idx()
        node_num = mlsl_obj.get_process_count()
        print 'rank ', node_idx
        print 'nodes ', node_num
        dist=mlsl_obj.create_distribution(node_num,1)
    else:
        mlsl_obj = None
        dist = None
    prefer_to_model_parallel = configuration['prefer_to_model_parallel']
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
    if devices:
        if prefer_to_model_parallel:
            enc_dec.build_trainer_with_model_parallel(src, src_mask, trg, trg_mask, ite, ps_device, devices, l1_reg_weight=l1_reg_weight, l2_reg_weight=l2_reg_weight)
        else:
            # clone the input
            src = [K.placeholder(shape=(None, None), dtype='int32') for _ in devices]
            src_mask = [K.placeholder(shape=(None, None)) for _ in devices]
            trg = [K.placeholder(shape=(None, None), dtype='int32') for _ in devices]
            trg_mask = [K.placeholder(shape=(None, None)) for _ in devices]

            enc_dec.build_trainer_with_data_parallel(src,
                                                     src_mask,
                                                     trg,
                                                     trg_mask,
                                                     ite,
                                                     devices,
                                                     l1_reg_weight=l1_reg_weight,
                                                     l2_reg_weight=l2_reg_weight,
                                                     softmax_output_num_sampled=softmax_output_num_sampled)
    else:
        enc_dec.build_trainer(src,
                              src_mask,
                              trg,
                              trg_mask,
                              ite,
                              l1_reg_weight=l1_reg_weight,
                              l2_reg_weight=l2_reg_weight,
                              softmax_output_num_sampled=softmax_output_num_sampled, mlsl_obj=mlsl_obj, dist=dist)

    enc_dec.build_sampler()

    if configuration['reload']:
        enc_dec.load()

    '''
    # comment for fast training
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
    '''

    # train function
    train_fn = enc_dec.train_fn
    
    # train data
    ds = DStream(**configuration)

    # valid data
    '''
    # comment for fast training
    vs = get_devtest_stream(data_type='valid', input_file=None, **configuration)
    '''
    
    iters = args.start
    valid_bleu_best = -1
    epoch_best = -1
    iters_best = -1
    max_epochs = configuration['finish_after']
    logger.info("epochs %d" %(max_epochs))
    fn = 'nmt_mkl_log'
    if mkl_multinode == True:
        if node_idx == 0:
            file = open(fn, 'w', 0)
            last_time = time.time()
            print('mkl multinode')
    else:
        file = open(fn, 'w', 0)
        last_time = time.time()       
        print('mkl single node') 
    for epoch in range(max_epochs):
        for x, x_mask, y, y_mask in ds.get_iterator():
            iter_count=0
            #last_time = time.time()
            # for data parallel, we need to split the data into #num devices part
            if devices and not prefer_to_model_parallel:
                # ignore the case that the number of samples is less than the number of devices
                num_devices = len(devices)
                num_samples = len(x)

                if num_samples < num_devices:
                    logger.warn('epoch %d \t updates %d ignored current mini-batch, since its number of samples (%d) < the number of devices (%d)'
                        % (epoch, iters, num_samples, num_devices))
                    continue

                inputs = []
                for data in (x, x_mask, y, y_mask):
                    parts = split(data, num_devices)
                    parts = [item.T for item in parts]
                    inputs.extend(parts)
            else:
                inputs = [x.T, x_mask.T, y.T, y_mask.T]
            #print('train start')
            tc = train_fn(inputs)
            #print('train finish')
            
            iters += 1

            #cur_time = time.time()
            #duration = cur_time - last_time
            #num_of_words = np.prod(x.shape)            
            #words_per_sec = int(num_of_words / duration)
            #logger.info('epoch %d \t updates %d train cost %.4f use time %.4f sec, %d words/sec, data x %s, data y %s'
            #            % (epoch, iters, tc[0], duration, words_per_sec, x.shape, y.shape))
            '''
            # Commented for fast training
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
            '''
            '''
            if mkl_multinode and node_idx == 0:
                file.write(str(tc[0])+'\n')
            else:
                file.write(str(tc[0])+'\n')
            '''
            iter_count+=1
    if mkl_multinode == True:
        if node_idx == 0:
            file.close()
            cur_time = time.time()
            duration = cur_time - last_time
            print('time one epoch ', duration)
    else:
        file.close()
        cur_time = time.time()
        duration = cur_time - last_time
        print('time one epoch ', duration)
    if mkl_multinode == True:
        mlsl_obj.delete_distribution(dist)
        mlsl_obj.finalize()    


if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",
                        default="get_config_search_coverage",
                        help="Prototype config to use for config")
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--start", type=int, default=0, help="Iterations to start")
    args = parser.parse_args()

    configuration = getattr(configurations, args.proto)()

    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    prefer_to_model_parallel = configuration['prefer_to_model_parallel']
    logger.info("Prefer_to_model_parallel %s" % prefer_to_model_parallel)

    ensure_vocabularies(**configuration)

    # update batch size accordingly if data parallel training is used.
    # Note that the first device is used as parameter server
    if devices:
        assert K._BACKEND == 'tensorflow'
        if not prefer_to_model_parallel:    # data parallel
            configuration['batch_size'] = (len(devices) - 1) * configuration['batch_size']
            logger.info("Batch size updated to %s" % configuration['batch_size'])
        ps_device = devices[0]
        wk_devices = devices[1:]
        with tf.device(ps_device):
            main(configuration, ps_device, wk_devices)
    else:
        main(configuration)
