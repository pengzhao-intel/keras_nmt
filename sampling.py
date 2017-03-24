# sampling: Sampler and BleuValidator
# from __future__ import print_function
import os
# os.environ["KERAS_BACKEND"] = "theano"

import numpy
import argparse
import pprint
import cPickle as pkl
import subprocess
import logging
import time
import re
import configurations
from search import BeamSearch
from nmt import EncoderDecoder
from data_stream import get_devtest_stream
from pinyin import get_pinyin

logger = logging.getLogger(__name__)


class Sampler(object):

    def __init__(self, search_model, **kwards):

        self.search_model = search_model
        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.hook_samples = kwards.pop('hook_samples')
        self.with_attention = kwards.pop('with_attention')
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)

    def apply(self, src_batch, trg_batch):

        batch_size = src_batch.shape[0]
        hook_samples = min(batch_size, self.hook_samples)
        sample_idx = numpy.random.choice(batch_size, hook_samples, replace=False)
        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.dict_src)
            target_length = self._get_true_length(target_[i], self.dict_trg)
            inp = input_[i, :input_length]
            results = self.search_model.apply(inp[:, None])
            outputs, costs = results[:2]
            if self.with_attention:
                if self.with_coverage:
                    coverages = results[3]
                    if self.coverage_type is 'linguistic':
                        fertilities = results[4]
            sample_length = self._get_true_length(numpy.array(outputs), self.dict_trg)

            logger.info("Input: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src)))
            logger.info("Target: {}".format(self._idx_to_word(target_[i][:target_length], self.idict_trg)))
            logger.info("Output: {}".format(self._idx_to_word(outputs[:sample_length], self.idict_trg)))
            if self.with_attention and self.with_coverage:
                logger.info("Coverage: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src, coverages)))
                if self.coverage_type is 'linguistic':
                    logger.info("Fertility: {}".format(self._idx_to_word(input_[i][:input_length], self.idict_src, fertilities)))
            logger.info("Cost: %.4f\n" % costs)


    def _get_dict(self, vocab_file):

        if os.path.isfile(vocab_file):
            ddict = pkl.load(open(vocab_file, 'rb'))
        else:
            logger.error("file [{}] do not exist".format(vocab_file))

        assert ddict

        iddict = dict()
        for kk, vv in ddict.iteritems():
            iddict[vv] = kk

        iddict[0] = self.eos_token

        return ddict, iddict


    def _get_true_length(self, seq, vocab):

        try:
            return seq.tolist().index(vocab[self.eos_token]) + 1
        except ValueError:
            return len(seq)

    def _idx_to_word(self, seq, ivocab, coverage=None):
        if coverage is None:
            return " ".join([ivocab.get(idx, self.unk_token).encode('utf-8') for idx in seq])
        else:
            output = []
            for _, [idx, ratio] in enumerate(zip(seq, coverage)):
                output.append('%s/%.2f' % (ivocab.get(idx, self.unk_token).encode('utf-8'), ratio))
            return " ".join(output)


class BleuValidator(object):

    def __init__(self, search_model, test_src=None, test_ref=None, **kwards):

        self.search_model = search_model
        self.unk_token = kwards.pop('unk_token')
        self.eos_token = kwards.pop('eos_token')
        self.vocab_src = kwards.pop('vocab_src')
        self.vocab_trg = kwards.pop('vocab_trg')
        self.normalize = kwards.pop('normalized_bleu')
        self.bleu_script = kwards.pop('bleu_script')
        self.res_to_sgm = kwards.pop('res_to_sgm')
        self.test_src = test_src
        self.test_ref = test_ref

        self.with_attention = kwards.pop('with_attention')
        self.output_kbest = kwards.pop('output_kbest')
        self.with_coverage = kwards.pop('with_coverage')
        self.coverage_type = kwards.pop('coverage_type')

        self.with_reconstruction = kwards.pop('with_reconstruction')

        # replace unk
        self.replace_unk = kwards.pop('replace_unk')
        if self.replace_unk:
            self.read_dict(kwards.pop('unk_dict'))

        if test_src is None or test_ref is None:
            self.test_src = kwards.pop('valid_src')
            self.test_ref = kwards.pop('valid_trg')

        self.dict_src, self.idict_src = self._get_dict(self.vocab_src)
        self.dict_trg, self.idict_trg = self._get_dict(self.vocab_trg)

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

    def replace_unk(self, source_words, output, alignment):
        tran_words = self._idx_to_word(output, self.idict_trg)
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

        return " ".join(new_tran_words)


    def apply(self, data_stream, out_file, verbose=False):
        logger.info("Begin decoding ...")
        fout = open(out_file, 'w')

        if self.output_kbest:
            fout_kbest = open(out_file + '.kbest', 'w')
        if self.replace_unk and self.with_attention:
            fout_runk = open(out_file + '.replaced.unk', 'w')
            if self.output_kbest:
                fout_kbest_runk = open(out_file + '.kbest.replaced.unk', 'w')

        val_start_time = time.time()
        i = 0
        for sent in data_stream.get_epoch_iterator():
            i += 1
            results = self.search_model.apply(numpy.array(sent).T)
            outputs, scores = results[:2]
            if self.with_attention:
                alignments = results[2]
                index = 3
                if self.with_coverage:
                    coverages = results[index]
                    index += 1
                    if self.coverage_type is 'linguistic':
                        fertilities = results[index]
                        index += 1

                if self.with_reconstruction:
                    reconstruction_scores = results[index]
                    inverse_alignments = results[index + 1]
                    index += 2

            if self.normalize:
                lengths = numpy.array([len(s) for s in outputs])
                scores = scores / lengths
            sidx = numpy.argmin(scores)
            res = self._idx_to_word(outputs[sidx][:-1], self.idict_trg)

            if res.strip() == '':
                res = self.unk_token

            fout.write(res + '\n')

            if self.replace_unk and self.with_attention:
                source_words = [self.idict_src.get(idx, self.unk_token) for idx in sent[0]]
                alignment = numpy.array(alignments[sidx]).transpose()
                print >> fout_runk, self.replace_unk(source_words, outputs[sidx][:-1], alignment)


            for idx in xrange(len(outputs)):
                kbest_score = [str(scores[idx])]
                aligns = [str(numpy.array(alignments[idx]).transpose().tolist())]
                if self.with_reconstruction:
                    kbest_score.extend([str(scores[idx] - reconstruction_scores[idx]), str(reconstruction_scores[idx])])
                    aligns.append(str(numpy.array(inverse_alignments[idx]).tolist()))

                if self.output_kbest:
                    print >> fout_kbest, '%d ||| %s ||| %s ||| %s' % (i, ' ||| '.join(kbest_score), self._idx_to_word(outputs[idx][:-1], self.idict_trg), ' ||| '.join(aligns))

                if self.replace_unk and self.with_attention:
                    alignment = numpy.array(alignments[idx]).transpose()
                    new_res = self.replace_unk(source_words, outputs[idx][:-1], alignment)
                    if self.output_kbest:
                        print >> fout_kbest_runk, '%d ||| %s ||| %s ||| %s' % (i, ' ||| '.join(kbest_score), new_res, ' ||| '.join(aligns))

            if verbose:
                # output alignment and coverage information
                print 'Translation:', res
                print 'Score:', scores[sidx]
                if self.with_attention:
                    print 'Aligns:'
                    print numpy.array(alignments[sidx]).transpose().tolist()
                    if self.with_coverage:
                        coverage = coverages[sidx]
                        # sent is a batch that contains only one sentence
                        sentence = [self.idict_src[idx] for idx in sent[0]]
                        print 'Coverage:',
                        for k in xrange(len(sentence)):
                            print '%s/%.2f' % (sentence[k], coverage[k]),
                        print ''
                        if self.coverage_type is 'linguistic':
                            print 'Fertility:',
                            for k in xrange(len(sentence)):
                                print '%s/%.2f' % (sentence[k], fertilities[k]),
                            print ''

                if self.with_reconstruction:
                    print 'Reconstruction Score:', reconstruction_scores[sidx]
                    print 'Inverse Aligns:'
                    print numpy.array(inverse_alignments[sidx]).tolist()

            if i % 100 == 0:
                logger.info("Translated {} lines of valid/test set ...".format(i))

        fout.close()

        logger.info("Decoding took {} minutes".format(float(time.time() - val_start_time) / 60.))

        logger.info("Evaluate ...")

        cmd_res_to_sgm = ['python', self.res_to_sgm, out_file, self.test_src + '.sgm', out_file + '.sgm']
        cmd_bleu_cmd = ['perl', self.bleu_script, \
                        '-r', self.test_ref + '.sgm', \
                        '-s', self.test_src + '.sgm', \
                        '-t', out_file + '.sgm', \
                        '>', out_file + '.eval']

        logger.info('covert result to sgm')
        subprocess.check_call(" ".join(cmd_res_to_sgm), shell=True)
        logger.info('compute bleu score')
        subprocess.check_call(" ".join(cmd_bleu_cmd), shell=True)

        fin = open(out_file + '.eval', 'rU')
        out = re.search('BLEU score = [-.0-9]+', fin.readlines()[7])
        fin.close()

        bleu_score = float(out.group()[13:])
        logger.info("Done")

        return bleu_score


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


    def _get_true_length(self, seq, vocab):

        try:
            return seq.tolist().index(vocab[self.eos_token]) + 1
        except ValueError:
            return len(seq)


    def _idx_to_word(self, seq, ivocab):

        return " ".join([ivocab.get(idx, self.unk_token) for idx in seq])


if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto",
                        default="get_config_search_coverage",
                        help="Prototype config to use for config")
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--beam", type=int, help="Beam size")
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('trans', type=str)
    args = parser.parse_args()

    configuration = getattr(configurations, args.proto)()

    if args.state:
        configuration.update(eval(open(args.state).read()))
    logger.info("\nModel options:\n{}".format(pprint.pformat(configuration)))

    enc_dec = EncoderDecoder(**configuration)
    enc_dec.build_sampler()

    if args.model:
        enc_dec.load(path=args.model)
    else:
        enc_dec.load(path=configuration['saveto_best'])

    beam_size = configuration['beam_size']
    if args.beam:
        beam_size = args.beam

    test_search = BeamSearch(enc_dec=enc_dec,
                             configuration=configuration,
                             beam_size=beam_size,
                             maxlen=3 * configuration['seq_len_src'], stochastic=False)
    bleuvalidator = BleuValidator(search_model=test_search,
                                  test_src=args.source,
                                  test_ref=args.target,
                                  **configuration)

    # test data
    ts = get_devtest_stream(data_type='test', input_file=args.source, **configuration)
    test_bleu = bleuvalidator.apply(ts, args.trans, True)

    logger.info('test bleu %.4f' % test_bleu)

