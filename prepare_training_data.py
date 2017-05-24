from picklable_itertools import iter_, chain
from fuel.utils.formats import open_

import logging
import random
import codecs

'''
Created on Apr 5, 2017

@author: lxh5147
'''

logger = logging.getLogger(__name__)

def prepare_training_data(src_files, src_files_encoding, trg_files, trg_files_encoding, src_output_file, trg_output_file):
    '''
    for each pair of source/target files, check they have the same number of sentences;
    do shuffle and save with utf-8 encodings
    '''
    src = chain(*[iter_(open_(f, encoding=src_files_encoding)) for f in src_files])
    trg = chain(*[iter_(open_(f, encoding=trg_files_encoding)) for f in trg_files])

    # TODO: find a way not to load all sentences into memory
    logger.info("reading sentences from source files...")
    src_sentences = [sent for sent in src]
    logger.info("reading sentences from target files...")
    trg_sentences = [sent for sent in trg]

    assert len(src_sentences) == len(trg_sentences)
    logger.info("number of sentences:%d" % len(src_sentences))

    # '\n' has been removed from a sentence
    assert  src_sentences[0].endswith('\n')
    # do the shuffle
    ids = list(range(len(src_sentences)))
    random.shuffle(ids)

    with codecs.open(src_output_file, 'w', 'UTF-8') as f_src:
        with codecs.open(trg_output_file, 'w', 'UTF-8') as f_trg:
            for i in ids:
                f_src.write(src_sentences[i])
                f_trg.write(trg_sentences[i])


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_files', required=True, help='source files, separated by ","')
    parser.add_argument('--source_files_encoding', default='UTF-8', help='encoding of source files')
    parser.add_argument('--target_files', required=True, help='target files, separated by ","')
    parser.add_argument('--target_files_encoding', default='UTF-8', help='encoding of target files')
    parser.add_argument('--source_output_file', required=True, help='source output file')
    parser.add_argument('--target_output_file', required=True, help='target output file')

    args = parser.parse_args()
    logger.info("run with args:%s" % args)
    prepare_training_data(src_files=args.source_files.split(','),
                          src_files_encoding=args.source_files_encoding,
                          trg_files=args.target_files.split(','),
                          trg_files_encoding=args.target_files_encoding,
                          src_output_file=args.source_output_file,
                          trg_output_file=args.target_output_file,)

