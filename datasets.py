'''
Created on Sep 4, 2016

@author: lxh5147
'''
# Refer to:  https://github.com/mila-udem/blocks-examples/blob/master/machine_translation/stream.py
from fuel.datasets import (TextFile, IterableDataset)
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)
from fuel.utils.formats import open_
import re
import os
import numpy as np
from picklable_itertools import izip

try:
    import cPickle as pickle
except:
    import pickle

def to_lower_case(line):
    line = line.lower()
    return line

# no bos; and eos is also used as padding
def _build_vocabulary(files, encoding,
                      eos='</S>',
                      eos_id=0,
                      unk='<UNK>',
                      unk_id=1,
                      max_nb_of_vacabulary=None,
                      preprocess=to_lower_case):
    stat = {}
    for filepath in files:
        with open_(filepath, encoding=encoding) as f:
            for line in f:
                if preprocess is not None:
                    line = preprocess(line)
                words = line.split()
                # replace number with NUM
                for word in words:
                    if word in stat:
                        stat[word] += 1
                    else:
                        stat[word] = 1
    sorted_items = sorted(stat.items(), key=lambda d: d[1], reverse=True)
    if max_nb_of_vacabulary is not None:
        sorted_items = sorted_items[:max_nb_of_vacabulary]
    vocab = {}
    vocab[eos] = eos_id
    vocab[unk] = unk_id
    special_token_idxs = set([ eos_id, unk_id])
    token_id = 0
    for token, _ in sorted_items:
        while token_id in special_token_idxs:
            token_id += 1
        vocab[token] = token_id
        token_id += 1
    return vocab


def _build_id_to_token_vocabulary(vocab):
    id_to_token_dict = {}
    for token, _id in vocab.items():
        id_to_token_dict[_id] = token
    return id_to_token_dict

def _save_vocabulary(vocab, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f, protocol=1)


def _load_vocabulary(filepath):
    return pickle.load(open(filepath))


class _length(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, target_source_index=-1):
        self.target_source_index = target_source_index
    def __call__(self, sentences):
        return len(sentences[self.target_source_index])


class _PaddingWithToken(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, pad_id, **kwargs):
        kwargs['data_stream'] = data_stream
        self.pad_id = pad_id
        super(_PaddingWithToken, self).__init__(**kwargs)

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [np.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = np.asarray(source_batch[0]).dtype

            padded_batch = np.zeros(
                (len(source_batch), max_sequence_length) + rest_shape,
                dtype=dtype) + self.pad_id
            for i, sample in enumerate(source_batch):
                padded_batch[i, :len(sample)] = sample
            batch_with_masks.append(padded_batch)

            mask = np.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self,
                 vocab_size=30000,
                 unk_id=1):
        self.vocab_size = vocab_size
        self.unk_id = unk_id

    def __call__(self, sentences):
        return tuple([[x if x < self.vocab_size else self.unk_id
                       for x in sentence]
                       for sentence in sentences])


class _remove_tokens(object):
    """Remove special tokens from a sentence."""
    def __init__(self, special_token_ids):
        self.special_token_ids = set(special_token_ids)

    def __call__(self, sentences):
        return tuple([[x for x in sentence if x not in self.special_token_ids]
                      for sentence in sentences])


class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentences):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentences])


class _to_one_hot(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, target_source_index, vacabuary_size):
        self.target_source_index = target_source_index
        self.vacabuary_size = vacabuary_size

    def __call__(self, data):
        x = data[self.target_source_index]
        x_shape = np.asarray(x).shape
        reshaped_x = np.asarray(x).flatten()
        len_x = len(reshaped_x)
        y = np.zeros((len_x, self.vacabuary_size))
        y[np.arange(len_x), reshaped_x] = 1
        return (np.reshape(y, x_shape + (self.vacabuary_size,)),)


class _to_token(object):
    """Converts id sequence to token sequence."""
    def __init__(self, id_to_token_vocab):
        self.id_to_token_vocab = id_to_token_vocab

    def __call__(self, id_sequence):
        return [self.id_to_token_vocab[_id] for _id in id_sequence]


def get_tr_stream(src_vocab,
                  trg_vocab,
                  src_files,
                  trg_files,
                  encoding='UTF-8',
                  preprocess=to_lower_case,
                  src_vocab_size=30000,
                  trg_vocab_size=30000,
                  eos='</S>',
                  eos_id=0,
                  unk='<UNK>',
                  unk_id=1,
                  seq_len=50,
                  batch_size=80,
                  sort_k_batches=12,
                  **kwargs):
    """Prepares the training data stream."""

    src_dataset = TextFile(src_files, src_vocab,
                           preprocess=preprocess,
                           bos_token=None,
                           eos_token=eos,
                           unk_token=unk,
                           encoding=encoding)
    trg_dataset = TextFile(trg_files, trg_vocab,
                           preprocess=preprocess,
                           bos_token=None,
                           eos_token=eos,
                           unk_token=unk,
                           encoding=encoding)

    src_data_stream = DataStream(src_dataset)
    trg_data_stream = DataStream(trg_dataset)


    # Replace out of vocabulary tokens with unk token
    if src_vocab_size < len(src_vocab):
        src_data_stream = Mapping(src_data_stream,
                         _oov_to_unk(vocab_size=src_vocab_size, unk_id=unk_id))

    if trg_vocab_size < len (trg_vocab):
        trg_data_stream = Mapping(trg_data_stream,
                                  _oov_to_unk(vocab_size=trg_vocab_size, unk_id=unk_id))

    # Merge them to get a source, target pair
    stream = Merge([src_data_stream, trg_data_stream],
                   ('source', 'target'))

    # Filter sequences that are too long (either source or target)
    stream = Filter(stream, predicate=_too_long(seq_len=seq_len))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length(target_source_index=1)))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    stream = _PaddingWithToken(stream, eos_id)

    # Attach one-hot ground truth data stream
    stream = Mapping(stream,
                     _to_one_hot(target_source_index=2, vacabuary_size=trg_vocab_size),
                     add_sources=("one_hot_ground_truth",))

    return stream


def _get_vl_stream(src_vocab,
                   trg_vocab,
                   src_files,
                   trg_files_list,
                   encoding='UTF-8',
                   preprocess=to_lower_case,
                   src_vocab_size=30000,
                   trg_vocab_size=30000,
                   eos='</S>',
                   eos_id=0,
                   unk='<UNK>',
                   unk_id=1,
                   batch_size=80,
                   sort_k_batches=12, **kwargs):
    """Prepares the validation/test data stream."""
    src_dataset = TextFile(src_files, src_vocab,
                           preprocess=preprocess,
                           bos_token=None,
                           eos_token=eos,
                           unk_token=unk,
                           encoding=encoding)

    trg_dataset_list = [TextFile(trg_files, trg_vocab,
                                 preprocess=preprocess,
                                 bos_token=None,
                                 eos_token=None,
                                 unk_token=unk,
                                 encoding=encoding)
                        for trg_files in trg_files_list]

    src_data_stream = DataStream(src_dataset)
    trg_data_stream_list = [DataStream(trg_dataset) for trg_dataset in trg_dataset_list]

    # Replace out of vocabulary tokens with unk token
    if src_vocab_size < len(src_vocab):
        src_data_stream = Mapping(src_data_stream,
                         _oov_to_unk(vocab_size=src_vocab_size,
                                     unk_id=unk_id))

    if trg_vocab_size < len (trg_vocab):
        trg_data_stream_list = [Mapping(trg_data_stream,
                         _oov_to_unk(vocab_size=trg_vocab_size,
                                     unk_id=unk_id)) for trg_data_stream in trg_data_stream_list]

    # Merge them to get a source, multiple references
    stream = Merge([src_data_stream] + trg_data_stream_list,
                   ('source',) + tuple(['reference_%d' % i for i in range(len(trg_data_stream_list))  ]))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length(target_source_index=0)))

    # Convert it into a stream again
    stream = Unpack(stream)

    # TODO: create dynamic batches, larger  batch size for shorter sentences, while smaller for longer sentence

    # Construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    stream = _PaddingWithToken(stream, eos_id, mask_sources=('source',))

    return stream


def _get_stream_from_lines(vocab,
                           lines,
                           preprocess=to_lower_case,
                           vocab_size=30000,
                           eos_id=0,
                           eos='</S>',
                           unk_id=1,
                           batch_size=80,
                           sort_k_batches=12):
    if preprocess is not None:
        lines = [preprocess(line) + ' ' + eos for line in lines]
    dataset = IterableDataset(iterables=lines)
    stream = DataStream(dataset)
    stream = Mapping(stream,
                 lambda x: ([ vocab[w] if w in vocab else unk_id for w in  x[0].split()],))

    if vocab_size < len(vocab):
        stream = Mapping(stream, _oov_to_unk(vocab_size=vocab_size, unk_id=unk_id))
    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * sort_k_batches))


    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length(target_source_index=0)))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    stream = _PaddingWithToken(stream, eos_id)

    return stream


def _get_iterator(stream):
    stream.reset()
    return stream.get_epoch_iterator()


def _get_generator_for_training(stream, use_sampled_softmax=False):
    iterator = _get_iterator(stream)
    while True:
        try:
            n = next(iterator)
            if use_sampled_softmax:
                yield [n[0], n[1], n[2], n[3]], n[2]    # ground_truth: id of ground truth
            else:
                yield [n[0], n[1], n[2], n[3]], n[-1]    # one-hot ground truth
        except StopIteration:
            iterator = _get_iterator(stream)

def _get_generator_for_testing(stream):
    iterator = _get_iterator(stream)
    while True:
        try:
            n = next(iterator)
            references = []
            references.extend(izip(*n[2:]))
            yield list(n[:2]), references
        except StopIteration:
            iterator = _get_iterator(stream)

def _get_src_sentences_with_references(stream):
    iterator = _get_iterator(stream)
    references = []
    n = next(iterator)
    references.extend(izip(*n[2:]))
    return list(n[:2]), references

def _get_generator_for_prediction(stream):
    iterator = _get_iterator(stream)
    while True:
        try:
            n = next(iterator)
            yield list(n[:2])
        except StopIteration:
            iterator = _get_iterator(stream)

def _get_src_sentences(stream):
    iterator = _get_iterator(stream)
    n = next(iterator)
    return list(n[:2])

def build_vocabulary_if_needed(files,
                               voc_filepath,
                               encoding='UTF-8',
                               eos='</S>',
                               eos_id=0,
                               unk='<UNK>',
                               unk_id=1,
                               max_nb_of_vacabulary=None,
                               preprocess=to_lower_case):
            # build vocabulary
        if os.path.isfile(voc_filepath):
            vocab = _load_vocabulary(voc_filepath)
        else:
            vocab = _build_vocabulary(files,
                                     encoding=encoding,
                                     eos=eos,
                                     eos_id=eos_id,
                                     unk=unk,
                                     unk_id=unk_id,
                                     max_nb_of_vacabulary=max_nb_of_vacabulary)
            _save_vocabulary(vocab, voc_filepath)

        return vocab

def get_generator_for_training(src_vocab,
                               trg_vocab,
                               src_files,
                               trg_files,
                               encoding='UTF-8',
                               preprocess=to_lower_case,
                               src_vocab_size=30000,
                               trg_vocab_size=30000,
                               eos='</S>',
                               eos_id=0,
                               unk='<UNK>',
                               unk_id=1,
                               seq_len=50,
                               batch_size=80,
                               sort_k_batches=12,
                               softmax_output_num_sampled=512,
                               **kwargs):

    stream = get_tr_stream(src_vocab=src_vocab,
                           trg_vocab=trg_vocab,
                           src_files=src_files,
                           trg_files=trg_files,
                           encoding=encoding,
                           preprocess=preprocess,
                           src_vocab_size=src_vocab_size,
                           trg_vocab_size=trg_vocab_size,
                           eos=eos,
                           eos_id=eos_id,
                           unk=unk,
                           unk_id=unk_id,
                           seq_len=seq_len,
                           batch_size=batch_size,
                           sort_k_batches=sort_k_batches,
                           **kwargs)

    return _get_generator_for_training(stream, use_sampled_softmax=softmax_output_num_sampled < trg_vocab_size)

def get_line_count(files, encoding='UTF-8'):
    count = 0
    for file_path in files:
        with open_(file_path, encoding=encoding) as f:
            for line in f:
                if line.rstrip('\n'):
                    count += 1
    return count

def get_src_sentences_with_references(src_vocab,
                                      trg_vocab,
                                      src_files,
                                      trg_files_list,
                                      encoding='UTF-8',
                                      preprocess=to_lower_case,
                                      src_vocab_size=30000,
                                      trg_vocab_size=30000,
                                      eos='</S>',
                                      eos_id=0,
                                      unk='<UNK>',
                                      unk_id=1,
                                      sort_k_batches=12, **kwargs):

    line_count = get_line_count (src_files, encoding)

    stream = _get_vl_stream(src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  src_files=src_files,
                  trg_files_list=trg_files_list,
                  encoding=encoding,
                  preprocess=preprocess,
                  src_vocab_size=src_vocab_size,
                  trg_vocab_size=trg_vocab_size,
                  eos=eos,
                  eos_id=eos_id,
                  unk=unk,
                  unk_id=unk_id,
                  batch_size=line_count,
                  sort_k_batches=sort_k_batches, **kwargs)

    outputs = _get_src_sentences_with_references(stream)
    stream.close()
    return outputs

def get_generator_for_testing(src_vocab,
                              trg_vocab,
                              src_files,
                              trg_files_list,
                              encoding='UTF-8',
                              preprocess=to_lower_case,
                              src_vocab_size=30000,
                              trg_vocab_size=30000,
                              batch_size=80,
                              eos='</S>',
                              eos_id=0,
                              unk='<UNK>',
                              unk_id=1,
                              sort_k_batches=12, **kwargs):

    stream = _get_vl_stream(src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  src_files=src_files,
                  trg_files_list=trg_files_list,
                  encoding=encoding,
                  preprocess=preprocess,
                  src_vocab_size=src_vocab_size,
                  trg_vocab_size=trg_vocab_size,
                  eos=eos,
                  eos_id=eos_id,
                  unk=unk,
                  unk_id=unk_id,
                  batch_size=batch_size,
                  sort_k_batches=sort_k_batches, **kwargs)

    return _get_generator_for_testing(stream)

def get_src_sentences(vocab,
                      lines,
                      preprocess=to_lower_case,
                      vocab_size=30000,
                      eos_id=0,
                      unk_id=1,
                      eos='</S>',
                      sort_k_batches=12):

    stream = _get_stream_from_lines(vocab=vocab,
              lines=lines ,
              preprocess=preprocess,
              vocab_size=vocab_size,
              eos_id=eos_id,
              unk_id=unk_id,
              eos=eos,
              batch_size=len(lines),
              sort_k_batches=sort_k_batches)

    outputs = _get_src_sentences(stream)
    stream.close()
    return outputs

def get_generator_for_prediction(vocab,
                                 lines,
                                 preprocess=to_lower_case,
                                 vocab_size=30000,
                                 batch_size=80,
                                 eos_id=0,
                                 unk_id=1,
                                 eos='</S>',
                                 sort_k_batches=12):

    stream = _get_stream_from_lines(vocab=vocab,
              lines=lines ,
              preprocess=preprocess,
              vocab_size=vocab_size,
              eos_id=eos_id,
              unk_id=unk_id,
              eos=eos,
              batch_size=batch_size,
              sort_k_batches=sort_k_batches)

    return _get_generator_for_prediction(stream)
