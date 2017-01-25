# data stream
import logging
import cPickle as pkl
import os
from fuel.datasets import TextFile
from fuel.streams import DataStream
from datasets import _lower_case_and_norm_numbers
from datasets import _get_tr_stream
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DStream(object):

    def __init__(self, **kwards):

        self.train_src = kwards.pop('train_src')
        self.train_trg = kwards.pop('train_trg')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.unk_token = kwards.pop('unk_token')
        self.unk_id = kwards.pop('unk_id')
        self.eos_token = kwards.pop('eos_token')
        self.seq_len_src = kwards.pop('seq_len_src')
        self.seq_len_trg = kwards.pop('seq_len_trg')
        self.batch_size = kwards.pop('batch_size')
        self.sort_k_batches = kwards.pop('sort_k_batches')

        # get source and target dicts
        self.src_dict, self.trg_dict = self._get_dict()
        self.eos_id = self.src_dict[self.eos_token]

        self.src_vocab_size = len(self.src_dict)
        self.trg_vocab_size = len (self.trg_dict)

        self.stream = _get_tr_stream(src_vocab=self.src_dict,
                      trg_vocab=self.trg_dict,
                      src_files=[self.train_src],
                      trg_files=[self.train_trg],
                      encoding='UTF-8',
                      preprocess=_lower_case_and_norm_numbers,
                      src_vocab_size=len(self.src_dict),
                      trg_vocab_size=len(self.trg_dict),
                      eos=self.eos_token,
                      eos_id=self.eos_id,
                      unk=self.unk_token,
                      unk_id=self.unk_id,
                      seq_len=80,
                      batch_size=self.batch_size,
                      sort_k_batches=self.sort_k_batches)



    def get_iterator(self):
        self.stream.reset()
        iterator = self.stream.get_epoch_iterator()
        for n in iterator:
            yield n[:4]    # ground_truth: id of ground truth, not one-hot representation


    def _get_dict(self):

        if os.path.isfile(self.vocab_src):
            src_dict = pkl.load(open(self.vocab_src, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_src))

        if os.path.isfile(self.vocab_trg):
            trg_dict = pkl.load(open(self.vocab_trg, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(self.vocab_trg))

        return src_dict, trg_dict

def get_devtest_stream(data_type='valid', input_file=None, **kwards):

    if data_type == 'valid':
        data_file = kwards.pop('valid_src')
    elif data_type == 'test':
        if input_file is None:
            data_file = kwards.pop('test_src')
        else:
            data_file = input_file
    else:
        logger.error('wrong datatype, which must be one of valid or test')

    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')
    vocab_src = kwards.pop('vocab_src')

    dataset = TextFile(files=[data_file],
                       encoding='UTF-8',
                       preprocess=_lower_case_and_norm_numbers,
                       dictionary=pkl.load(open(vocab_src, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    dev_stream = DataStream(dataset)

    return dev_stream

def get_stream(input_file, vocab_file, **kwards):
    unk_token = kwards.pop('unk_token')
    eos_token = kwards.pop('eos_token')

    dataset = TextFile(files=[input_file],
                       encoding='UTF-8',
                       preprocess=_lower_case_and_norm_numbers,
                       dictionary=pkl.load(open(vocab_file, 'rb')),
                       level='word',
                       unk_token=unk_token,
                       bos_token=None,
                       eos_token=eos_token)

    stream = DataStream(dataset)

    return stream

