#!/usr/bin/env python
# encoding=utf-8

import argparse
import pprint
import logging

import time
import os
import numpy
import cPickle as pkl

import configurations as configurations
from search import BeamSearch
from nmt import EncoderDecoder
from pinyin import get_pinyin

import BaseHTTPServer
from subprocess import Popen, PIPE
import urllib
from preprocess import preprocess

from eai_arkseg.arksegmentor import PyMetaSeg

seg_model = PyMetaSeg.PyMetaModel(os.path.abspath('./eai_arkseg') + "/libarkseg.so", \
                                  os.path.abspath('./eai_arkseg') + "/data/dict.bin", \
                                  os.path.abspath('./eai_arkseg') + "/data/model.bin", \
                                  os.path.abspath('./eai_arkseg') + "/data/user_dict.txt", \
                                  os.path.abspath('./eai_arkseg') + "/data/ambi_dict.txt", \
                                  0)
segmentor = PyMetaSeg.PyMetaSegmentor(seg_model, 400)


def a2b(input):
    # a->b
    ustring = unicode(input, 'utf8')
    output = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # special process for a blank
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        output += unichr(inside_code)

    return output.encode('utf8')


# break into sentences
def break_into_sentences(paragraph, sentsinfo, sents):
    count = 0
    for i in xrange(len(paragraph)):
        res = preprocess.break_into_sentences(paragraph[i])
        for sent in res:
            sents.append(sent.strip())
            sentsinfo[count] = i
            count += 1


# restore paragraph
def restore_paragraph(sents, sentsinfo):
    content = sents[0] + ' '
    for i in xrange(1, len(sents)):
        if sentsinfo[i - 1] != sentsinfo[i]:
            content += '\n'

        content += sents[i] + ' '
    return content.strip()


def break_into_words(sents):
    # word breaker
    for i in xrange(len(sents)):
        sents[i] = ' '.join(segmentor.segment(sents[i].decode('utf-8'))).encode('utf-8')


class MTReqHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        logger.info('header:%s' % self.headers)
        logger.info('path:%s' % self.path)

        args = self.path.split('?')[1]
        args = args.split('&')
        source = None
        ignore_unk = False
        beamwidth = 10
        for aa in args:
            cc = aa.split('=')
            if len(cc) == 0:
                logger.warning('Ignored invalid arg:%s' % aa)
                continue
            if cc[0] == 'source':
                source = cc[1]
            if cc[0] == 'ignore_unk':
                ignore_unk = int(cc[1])
            if cc[0] == 'beamwidth':
                beamwidth = int(cc[1])

        if source is None:
            self.send_response(400)
            return

        source = a2b(urllib.unquote_plus(source))
        logger.info('source:%s' % source)
        # break into paragraphs
        paragraph = source.split('\n')

        sentsinfo = dict()  # sentence to paragraph
        sents = list()
        # break into sentences
        break_into_sentences(paragraph, sentsinfo, sents)
        start_time = time.time()
        # break into words
        break_into_words(sents)
        logger.info('Time used by segmenter:%s' % str(time.time() - start_time))

        start_time = time.time()
        trans = list()
        unknown_words = list()
        for sent in sents:
            if len(sent) == 0:
                tran = ''
                unks = []
            else:
                tran, unks = self.server.sampler.sample(sent,
                                                        ignore_unk=ignore_unk,
                                                        beamwidth=beamwidth)
            trans.append(tran.strip())
            unknown_words.extend(unks)

        logger.info('Time used by translator:%s' % str(time.time() - start_time))

        translation = restore_paragraph(trans, sentsinfo)
        logger.info('translation:%s' % translation)

        unknown_words = list(set(unknown_words))

        response = urllib.quote_plus(translation + '||||||' + ' ||| '.join(unknown_words))

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response)


class Sampler:
    def __init__(self, search_model, **kwards):
        self.search_model = search_model
        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')

        self.with_attention = kwards.pop('with_attention')
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')
        self.replace_unk = kwards.pop('replace_unk')

        if self.replace_unk:
            self.read_dict(kwards.pop('unk_dict'))

        # TODO: move the script to a separated folder
        tokenizer_cmd = [os.getcwd() + '/tokenizer.perl', '-l', 'en', '-q', '-']
        detokenizer_cmd = [os.getcwd() + '/detokenizer.perl', '-l', 'en', '-q', '-u', '-']

        self.tokenizer_cmd = tokenizer_cmd
        self.detokenizer_cmd = detokenizer_cmd

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)

        self.unk_id = self.dict_src[self.unk_token]
        self.eos_id = self.dict_src[self.eos_token]

        self.normalize = True

    def read_dict(self, dict_file):
        self.unk_dict = {}
        fin = open(dict_file)
        while 1:
            try:
                line = fin.next().strip()
            except StopIteration:
                break

            src, tgt = line.split()
            self.unk_dict[src] = tgt

    def sample(self, sentence, ignore_unk=False, beamwidth=10):
        ids = self._word_to_idx(sentence, self.dict_src)

        results = self.search_model.apply(numpy.array([ids]).T)
        outputs, scores = results[:2]
        if self.with_attention:
            alignments = results[2]

        if self.normalize:
            lengths = numpy.array([len(s) for s in outputs])
            scores = scores / lengths
        sidx = numpy.argmin(scores)
        res = self._idx_to_word(outputs[sidx][:-1], self.idict_trg)

        translated_unks = set()

        if self.replace_unk and self.with_attention:
            source_words = sentence.split() + [self.eos_token]
            tran_words = res.split()
            alignment = numpy.array(alignments[sidx]).transpose()
            # get the hard alignment
            aligned_source_words = [source_words[idx] for idx in numpy.argmax(alignment, axis=0)]
            new_tran_words = []
            for i in xrange(len(tran_words)):
                if tran_words[i] != self.unk_token:
                    new_tran_words.append(tran_words[i])
                else:
                    # replace unk token
                    aligned_source_word = aligned_source_words[i]
                    # note that get_pinyin only accept Chinese word in GBK encoding
                    new_tran_words.append(self.unk_dict.get(aligned_source_word, get_pinyin(aligned_source_word)))
                    if aligned_source_word in self.unk_dict:
                        translated_unks.add(aligned_source_word)

            logger.info('new_tran_words:%s' % new_tran_words)
            res = " ".join(new_tran_words)

        if self.detokenizer_cmd:
            detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
            res, _ = detokenizer.communicate(res)

        unknown_words = [word for word, index
                         in zip(sentence.split(), ids)
                         if index == self.unk_id and word not in translated_unks]

        return res, unknown_words

    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token

        return ddict, iddict

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, self.unk_token) for idx in seq])

    def _word_to_idx(self, seq, vocab):
        return [vocab.get(w, self.unk_id) for w in seq.strip().split()] + [self.eos_id]


def parse_args():
    parser = argparse.ArgumentParser(
        "Sample (of find with beam-serch) translations from a translation model")
    parser.add_argument("--port", help="Port to use", type=int, default=8888)
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--beam-search",
                        action="store_true", help="Beam size, turns on beam-search")
    parser.add_argument("--beam-size",
                        type=int, help="Beam size", default=5)
    parser.add_argument("--ignore-unk",
                        default=False, action="store_true",
                        help="Ignore unknown words")
    parser.add_argument("--normalize",
                        action="store_true", default=False,
                        help="Normalize log-prob with the word count")
    parser.add_argument("model_path",
                        help="Path to the model")
    return parser.parse_args()


# start the server:LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib/ python ./sample_server.py --beam-search --state ./config.py --port 8888 data/model.npz
# run the test: curl http://localhost:8888?source=我要来

if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, MTReqHandler)

    configuration = getattr(configurations, 'get_config_search_coverage')()

    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    enc_dec = EncoderDecoder(**configuration)
    enc_dec.build_sampler()

    enc_dec.load(path=args.model_path)

    search = BeamSearch(enc_dec=enc_dec,
                        beam_size=args.beam_size,
                        maxlen=3 * configuration['seq_len_src'],
                        stochastic=False,
                        configuration=configuration)

    sampler = Sampler(search, **configuration)
    httpd.sampler = sampler

    logger.info('Server starting..')
    httpd.serve_forever()
