# models: BeamSearch
import numpy
import copy


class BeamSearch(object):

    def __init__(self, enc_dec, configuration, beam_size=1, maxlen=50, stochastic=True):
        # with_attention=True, with_coverage=False, coverage_dim=1, coverage_type='linguistic', max_fertility=2, with_reconstruction=False, reconstruction_weight, with_reconstruction_error_on_states=False,):

        self.enc_dec = enc_dec
        # if sampling, beam_size = 1
        self.beam_size = beam_size
        # max length of output sentence
        self.maxlen = maxlen
        # stochastic == True stands for sampling
        self.stochastic = stochastic

        self.with_attention = configuration['with_attention']

        self.with_coverage = configuration['with_coverage']
        self.coverage_dim = configuration['coverage_dim']
        self.coverage_type = configuration['coverage_type']
        self.max_fertility = configuration['max_fertility']

        self.with_reconstruction_coverage = configuration['with_reconstruction_coverage']

        self.with_reconstruction = configuration['with_reconstruction']
        self.reconstruction_weight = configuration['reconstruction_weight']
        self.with_reconstruction_error_on_states = configuration['with_reconstruction_error_on_states']
        self.with_attention_agreement = configuration['with_attention_agreement']
        self.attention_agreement_weight = configuration['attention_agreement_weight']


        if self.beam_size > 1:
            assert not self.stochastic, 'Beam search does not support stochastic sampling'


    def apply(self, input):
        sample = []
        sample_score = []
        if self.stochastic:
            sample_score = 0
        # for reconstruction
        sample_states = []
        if self.with_attention:
            sample_alignment = []
            if self.with_coverage:
                sample_coverage = []

        # get initial state of decoder rnn and encoder context
        ret = self.enc_dec.compile_init_and_context([input])
        next_state, c = ret[0], ret[1]
        if not self.with_attention:
            init_ctx0 = ret[2]

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = [[]] * live_k

        if self.with_attention:
            hyp_alignments = [[]]
            # note that batch size is the second dimension coverage and will be used in the later decoding, thus we need a structure different from the above ones
            if self.with_coverage:
                hyp_coverages = numpy.zeros((c.shape[0], 1, self.coverage_dim), dtype='float32')
                if self.coverage_type is 'linguistic':
                    # note the return result is a list even when it contains only one element
                    fertility = self.enc_dec.compile_fertility([c])[0]
        # bos indicator
        next_w = -1 * numpy.ones((1,)).astype('int32')

        for i in range(self.maxlen):
            inps = [next_w, next_state]
            if self.with_attention:
                ctx = numpy.tile(c, [live_k, 1])
                inps.append(ctx)
                if self.with_coverage:
                    inps.append(hyp_coverages)
            else:
                init_ctx = numpy.tile(init_ctx0, [live_k, 1])
                inps.append(init_ctx)

            ret = self.enc_dec.compile_next_state_and_probs(inps)
            next_p, next_state, next_w = ret[0], ret[1], ret[2]
            if self.with_attention:
                alignment = ret[3]
                # update the coverage after attention operation
                if self.with_coverage:
                    coverages = ret[4]

            if self.stochastic:
                nw = next_w[0]
                sample.append(nw)
                sample_states.append(next_state[0])
                sample_score += next_p[0, nw]
                if self.with_attention and self.with_coverage:
                    hyp_coverages = coverages
                # 0 for EOS
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:, None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:self.beam_size - dead_k]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = numpy.zeros(self.beam_size - dead_k).astype('float32')
                new_hyp_states = []
                if self.with_attention:
                    new_hyp_alignments = []
                    if self.with_coverage:
                        new_hyp_coverages = numpy.zeros((c.shape[0], self.beam_size - dead_k, self.coverage_dim), dtype='float32')

                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(hyp_states[ti] + [copy.copy(next_state[ti])])
                    if self.with_attention:
                        new_hyp_alignments.append(hyp_alignments[ti] + [alignment[:, ti]])
                        if self.with_coverage:
                            new_hyp_coverages[:, idx, :] = coverages[:, ti, :]

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []
                if self.with_attention:
                    hyp_alignments = []
                    if self.with_coverage:
                        indices = []

                for idx in range(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        if self.with_attention:
                            # for reconstruction
                            sample_states.append(new_hyp_states[idx])
                            sample_alignment.append(new_hyp_alignments[idx])
                            if self.with_coverage:
                                # for neural coverage, we use the mean value of the vector
                                sample_coverage.append(new_hyp_coverages[:, idx, :].mean(1))

                        dead_k += 1
                    else:
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        if self.with_attention:
                            hyp_alignments.append(new_hyp_alignments[idx])
                            if self.with_coverage:
                                indices.append(idx)

                        new_live_k += 1

                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k

                if self.with_attention:
                    if self.with_coverage:
                        # note now liv_k has changed
                        hyp_coverages = numpy.zeros((c.shape[0], live_k, self.coverage_dim), dtype='float32')
                        for idx in xrange(live_k):
                            hyp_coverages[:, idx, :] = new_hyp_coverages[:, indices[idx], :]

                if live_k < 1 or dead_k >= self.beam_size:
                    break

                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = numpy.array([s[-1] for s in hyp_states])

        if not self.stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
                    sample_states.append(hyp_states[idx])
                    if self.with_attention:
                        sample_alignment.append(hyp_alignments[idx])
                        if self.with_coverage:
                            sample_coverage.append(hyp_coverages[:, idx, :].mean(1))
        else:
            if self.with_attention and self.with_coverage:
                sample_coverage = hyp_coverages[:, 0, :].mean(1)

        # for reconstruction
        if self.with_reconstruction:
            # build inverce_c and mask
            if self.stochastic:
                sample_states = [sample_states]
            sample_num = len(sample_states)

            inverse_sample_score = numpy.zeros(sample_num).astype('float32')
            if self.with_attention:
                inverse_sample_alignment = [[] for i in  xrange(sample_num)]

            my = max([len(s) for s in sample_states])
            inverse_c = numpy.zeros((my, sample_num, sample_states[0][0].shape[0]), dtype='float32')
            mask = numpy.zeros((my, sample_num), dtype='float32')
            for idx in range(sample_num):
                inverse_c[:len(sample_states[idx]), idx, :] = sample_states[idx]
                mask[:len(sample_states[idx]), idx] = 1.

            # get initial state of decoder rnn and encoder context
            inverse_ret = self.enc_dec.compile_inverse_init_and_context([inverse_c])
            inverse_next_state = inverse_ret[0]
            if not self.with_attention:
                inverse_init_ctx0 = inverse_ret[1]

            if self.with_attention:
                # note that batch size is the second dimension coverage and will be used in the later decoding, thus we need a structure different from the above ones
                if self.with_reconstruction_coverage:
                    inverse_hyp_coverages = numpy.zeros((inverse_c.shape[0], sample_num, self.coverage_dim), dtype='float32')

            to_reconstruct_input = input[:, 0]
            for i in range(len(to_reconstruct_input)):
                # whether input contains eos?
                inverse_next_w = numpy.array([to_reconstruct_input[i - 1]] * sample_num) if i > 0 else -1 * numpy.ones((sample_num,)).astype('int32')

                inps = [inverse_next_w, mask, inverse_next_state]
                if self.with_attention:
                    inps.append(inverse_c)
                    if self.with_reconstruction_coverage:
                        inps.append(inverse_hyp_coverages)
                else:
                    inps.append(inverse_init_ctx0)

                ret = self.enc_dec.compile_inverse_next_state_and_probs(inps)
                inverse_next_p, inverse_next_state, inverse_next_w = ret[0], ret[1], ret[2]
                if self.with_attention:
                    inverse_alignment = ret[3]
                    # update the coverage after attention operation
                    if self.with_reconstruction_coverage:
                        inverse_hyp_coverages = ret[4]

                # compute reconstruction error
                if self.with_reconstruction_error_on_states:
                    # calculate Euclidean distance
                    inverse_sample_score += [numpy.linalg.norm(c[i, 0, :] - reconstructed_state) for reconstructed_state in inverse_next_state]
                else:
                    inverse_sample_score -= numpy.log(inverse_next_p[:, to_reconstruct_input[i]])

                # for each sample
                for idx in range(sample_num):
                    inverse_sample_alignment[idx].append(inverse_alignment[:len(sample_states[idx]), idx])

            # combine sample_score and reconstructed_score
            sample_score += inverse_sample_score * self.reconstruction_weight

            if self.with_attention_agreement:
                sample_mul_score = numpy.zeros(sample_num).astype('float32')
                for i in xrange(len(sample_alignment)):
                    mul_cost = -numpy.log(numpy.sum(numpy.array(sample_alignment[i]) * (numpy.array(inverse_sample_alignment[i]).transpose())) + 1e-6)
                    sample_mul_score[i] = mul_cost

                sample_score += sample_mul_score * self.attention_agreement_weight


        results = [sample, sample_score]
        if self.with_attention:
            results.append(sample_alignment)
            if self.with_coverage:
                results.append(sample_coverage)
                if self.coverage_type is 'linguistic':
                    results.append(fertility[:, 0])

            if self.with_reconstruction:
                results.append(inverse_sample_score)
                results.append(inverse_sample_alignment)

                if self.with_attention_agreement:
                    results.append(sample_mul_score)

        return results

# for reconstructing inputs from a given generation
class Reconstruct(object):

    def __init__(self, enc_dec, configuration, beam_size=1, maxlen=50, stochastic=True):
        # with_attention=True, with_coverage=False, coverage_dim=1, coverage_type='linguistic', max_fertility=2, with_reconstruction=False, reconstruction_weight, with_reconstruction_error_on_states=False,):

        self.enc_dec = enc_dec
        # if sampling, beam_size = 1
        self.beam_size = beam_size
        # max length of output sentence
        self.maxlen = maxlen
        # stochastic == True stands for sampling
        self.stochastic = stochastic

        self.with_attention = configuration['with_attention']
        self.with_coverage = configuration['with_coverage']
        self.coverage_dim = configuration['coverage_dim']
        self.coverage_type = configuration['coverage_type']
        self.max_fertility = configuration['max_fertility']

        self.with_reconstruction_coverage = configuration['with_reconstruction_coverage']
        self.with_reconstruction = configuration['with_reconstruction']
        self.reconstruction_weight = configuration['reconstruction_weight']
        self.with_reconstruction_error_on_states = configuration['with_reconstruction_error_on_states']
        self.with_attention_agreement = configuration['with_attention_agreement']
        self.attention_agreement_weight = configuration['attention_agreement_weight']


        if self.beam_size > 1:
            assert not self.stochastic, 'Beam search does not support stochastic sampling'


    def apply(self, source, target):
        alignment = []

        # get initial state of decoder rnn and encoder context
        ret = self.enc_dec.compile_init_and_context([source])
        next_state, ctx = ret[0], ret[1]

        decoder_states = numpy.zeros((target.shape[0], 1, next_state.shape[1]), dtype='float32')

        if self.with_coverage:
            coverage = numpy.zeros((ctx.shape[0], 1, self.coverage_dim), dtype='float32')
            if self.coverage_type is 'linguistic':
                # note the return result is a list even when it contains only one element
                fertility = self.enc_dec.compile_fertility([ctx])[0]

        for i in range(len(target)):
            next_w = numpy.array(target[i - 1]) if i > 0 else -1 * numpy.ones((1,)).astype('int32')
            inps = [next_w, next_state, ctx]

            if self.with_coverage:
                inps.append(coverage)
                if self.coverage_type is 'linguistic':
                    inps.append(fertility)

            ret = self.enc_dec.compile_next_state_and_probs(inps)
            _, next_state, next_w, align = ret[0], ret[1], ret[2], ret[3]
            # update the coverage after attention operation
            if self.with_coverage:
                coverage = ret[4]

            alignment.append(align[:, 0])
            decoder_states[i, 0, :] = next_state[0, :]

        # for reconstructing source
        reconstructed_source = []
        reconstructed_alignment = []
        reconstructed_score = 0.

        inverse_c = decoder_states
        mask = numpy.ones((target.shape[0], 1), dtype='float32')

        inverse_ret = self.enc_dec.compile_inverse_init_and_context([inverse_c])
        inverse_next_state = inverse_ret[0]

        # bos indicator
        inverse_next_w = -1 * numpy.ones((1,)).astype('int32')

        for i in range(self.maxlen):
            inps = [inverse_next_w, mask, inverse_next_state, inverse_c]

            ret = self.enc_dec.compile_inverse_next_state_and_probs(inps)
            inverse_next_p, inverse_next_state, inverse_next_w, inverse_alignment = ret[0], ret[1], ret[2], ret[3]

            # we reconstruct the input in stochastic mode
            inverse_nw = inverse_next_w[0]
            reconstructed_source.append(inverse_nw)
            reconstructed_alignment.append(inverse_alignment[:, 0])
            reconstructed_score += inverse_next_p[0, inverse_nw]

            # 0 for eos indicator
            if inverse_nw == 0:
                break

        return reconstructed_source, reconstructed_score, alignment, reconstructed_alignment

# for forced alignment
class Align(object):
    def __init__(self, enc_dec, with_attention=True, with_coverage=False, coverage_dim=1, coverage_type='linguistic', max_fertility=2, with_reconstruction=False, with_reconstruction_coverage=False):
        self.enc_dec = enc_dec

        assert with_attention, "Align only supports attention model"

        self.with_coverage = with_coverage
        self.coverage_dim = coverage_dim
        self.coverage_type = coverage_type
        self.max_fertility = max_fertility
        self.with_reconstruction = with_reconstruction
        self.with_reconstruction_coverage = with_reconstruction_coverage

    def apply(self, source, target):
        alignment = []

        # get initial state of decoder rnn and encoder context
        ret = self.enc_dec.compile_init_and_context([source])
        next_state, ctx = ret[0], ret[1]

        decoder_states = numpy.zeros((target.shape[0], 1, next_state.shape[1]), dtype='float32')

        if self.with_coverage:
            coverage = numpy.zeros((ctx.shape[0], 1, self.coverage_dim), dtype='float32')
            if self.coverage_type is 'linguistic':
                # note the return result is a list even when it contains only one element
                fertility = self.enc_dec.compile_fertility([ctx])[0]

        for i in range(len(target)):
            next_w = numpy.array(target[i - 1]) if i > 0 else -1 * numpy.ones((1,)).astype('int32')
            inps = [next_w, next_state, ctx]

            if self.with_coverage:
                inps.append(coverage)
                if self.coverage_type is 'linguistic':
                    inps.append(fertility)

            ret = self.enc_dec.compile_next_state_and_probs(inps)
            _, next_state, next_w, align = ret[0], ret[1], ret[2], ret[3]
            # update the coverage after attention operation
            if self.with_coverage:
                coverage = ret[4]

            alignment.append(align[:, 0])
            decoder_states[i, 0, :] = next_state[0, :]

        if self.with_reconstruction:
            inverse_alignment = []

            ret = self.enc_dec.compile_inverse_init_and_context([decoder_states])
            inverse_next_state = ret[0]

            if self.with_reconstruction_coverage:
                inverse_coverage = numpy.zeros((decoder_states.shape[0], 1, self.coverage_dim), dtype='float32')
                if self.coverage_type is 'linguistic':
                    inverse_fertility = self.enc_dec.compile_inverse_fertility([decoder_states])[0]

            mask = numpy.ones((decoder_states.shape[0], 1), dtype='float32')

            for i in range(len(source)):
                inverse_next_w = numpy.array(source[i - 1]) if i > 0 else -1 * numpy.ones((1,)).astype('int32')
                inps = [inverse_next_w, mask, inverse_next_state, decoder_states]
                if self.with_reconstruction_coverage:
                    inps.append(inverse_coverage)
                    if self.coverage_type is 'linguistic':
                        inps.append(inverse_fertility)

                ret = self.enc_dec.compile_inverse_next_state_and_probs(inps)
                _, inverse_next_state, inverse_next_w, inverse_align = ret[0], ret[1], ret[2], ret[3]
                if self.with_reconstruction_coverage:
                    inverse_coverage = ret[4]

                inverse_alignment.append(inverse_align[:, 0])

        results = [alignment]
        if self.with_coverage:
            coverage = coverage[:, 0, :].mean(1)
            results.append(coverage)
            if self.coverage_type is 'linguistic':
                results.append(fertility[:, 0])

        if self.with_reconstruction:
            results.append(inverse_alignment)
            if self.with_reconstruction_coverage:
                inverse_coverage = inverse_coverage[:, 0, :].mean(1)
                results.append(inverse_coverage)
                if self.coverage_type is 'linguistic':
                    results.append(inverse_fertility[:, 0])

        return results

