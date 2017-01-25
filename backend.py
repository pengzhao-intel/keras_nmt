'''
Created on Jul 16, 2016

@author: lxh5147
'''
from keras import backend as K
from keras.backend.common import _FLOATX
import numpy as np
def repeat(x, n):
    '''Repeats a tensor along the first dimension:
    For example, if x has shape (samples, dim) and n=2,
    the output will have shape (samples*2, dim)

    # Parameters
    ----------
    x : a tensor
    n: times to repeat

    # Returns
    ------
    the repeated tensor

    '''
    x_shape = K.shape(x)
    x_ndim = K.ndim(x)
    # to 1D tensor
    x_tiled = K.tile(K.reshape(x, (-1,)), n)
    # re-shape to (n,...)
    x_tiled_shape = pack([n] + [x_shape[i] for i in range(x_ndim)])
    output = K.reshape(x_tiled, x_tiled_shape)
    pattern = [1, 0] + [i + 1 for i in range(1, x_ndim)]
    output = K.permute_dimensions(output, pattern)
    output_shape = pack([n * x_shape[0]] + [x_shape[i] for i in range(1, x_ndim)])
    return K.reshape(output, output_shape)

def get_shape(x):
    '''Gets shape (i.e., a tuple of integers) of a keras tensor.

    # Parameters
    ----------
    x : a keras tensor, which has a property of _keras_shape

    # Returns
    ------
    a tuple of integers, which representing the get_shape of the keras tensor

    # Raises
    ------
    Exception
        if the input tensor does not has _keras_shape property
    '''
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise Exception('You tried to call get_shape on a tensor without information about its expected input get_shape.')

def get_mask(x, mask_value=0):
    '''Returns a boolean tensor, representing the mask of x. The mask of a tensor has the same shape of that tensor,
    and further given a position, the mask at that position is True if and only if the value of that tensor at that
    position does not equal to mask_value.

    For example, given x=[1,2,3,3], mask_value=3, the mask of x is [1,1,0,0]

    # Parameters
    ----------
    x : a tensor
    mask_value: mask value

    # Returns
    ------
    the mask of that tensor

    '''
    return K.not_equal (x, mask_value)

def get_vector_mask(x, mask_value=0):
    return K.any(K.not_equal(x, mask_value), axis=-1)

def apply_mask(input_tensor, input_mask):
    assert K.ndim(input_tensor) >= 2
    assert input_mask is None \
            or K.ndim(input_mask) == K.ndim(input_tensor) \
            or K.ndim(input_mask) == K.ndim(input_tensor) - 1 \
            or K.ndim(input_mask) == K.ndim(input_tensor) + 1
    if input_mask is None:
        return input_tensor
    else:
        # input_tensor: nb_samples,..., time_steps, input_dim
        # input_mask: nb_samples, ..., time_steps
        if K.ndim(input_mask) == K.ndim(input_tensor) - 1:
            mask = K.expand_dims(input_mask)
        elif K.ndim(input_mask) == K.ndim(input_tensor) + 1:
            mask = K.squeeze(input_mask, K.ndim(input_mask) - 1)
        else:
            mask = input_mask
        mask = K.cast(mask, K.dtype(input_tensor))
        return input_tensor * mask

def get_time_step_length_without_padding(x, time_step_dim=-2, padding=0):
    '''Gets time steps without padding (right) of a input tensor.

    # Parameters
    ----------
    x : a tensor whose dimensions >=3
    time_step_dim: the time step dimension of x
    padding : a scalar tensor that represents the padding
    # Returns
    ------
    a tensor represents the length of the input tensor after removing all right padding zeros
    '''
    ndim = K.ndim(x)
    time_step_dim = time_step_dim % ndim
    x = K.cast(K.not_equal(x, padding), 'int32')    # binary tensor
    axis = [i for i in range(ndim) if i != time_step_dim]
    s = K.sum(x, axis)
    s = K.cast(K.not_equal(s, 0), 'int32')
    return K.sum(s)

def inner_product(x, y):
    '''Gets the inner product between a tensor and a vector. The last dimension of that tensor must have the same shape as the vector.

    # Parameters
    ----------
    x : a tensor whose dimensions >=2, of a shape .., vector_dim
    y : a vector (one dimension vector of shape vector_dim

    # Returns
    ------
    a tensor with ndim-1 dimensions, where ndim is the number of dimensions of the input tensor
    '''
    ndim = K.ndim(x)
    dim = -2 if ndim > 1 else 0
    x = K.expand_dims(x, dim)    # ..., 1, vector_dim
    y = K.expand_dims(y)    # vector_dim,1
    output = dot(x, y)    # ..., 1*1
    output = K.squeeze(output, -1)
    output = K.squeeze(output, -1)
    return output

def beam_search_first_step(_step_score, _state, embedding, beam_size):
    '''Defines internal helper function that executes the first step for beam search.

    # Parameters
    ----------
    _step_score : a tensor representing score after executing the first step, which has a shape number_samples, output_dim
    _state : a tensor of a shape of nb_samples, state_dim, representing the state after executing the first step
    embedding : an embedding layer that maps input/output labels to their embedding
    beam_size: beam size

    # Returns
    ------
    top_score: a tensor with a shape of nb_samples, beam_size, representing scores of top beam_size predictions
    output_label_id: a tensor with a shape of nb_samples, beam_size, representing the labels of the top beam_size predictions
    prev_output_index: a tensor with a shape of nb_samples, beam_size, with zero as its value
    current_input: a tensor with a shape of nb_samples*beam_size, input_dim, corresponding to the embedding of output_label_id (reshaped)
    current_state: a tensor with a shape of nb_samples*beam_size; the beam_size candidates derived from the i^th sample has a state _state[i]
    current_score: a tensor with a shape of nb_samples*beam_size, reshaped version of top_score
    '''
    top_score , top_indice = top_k (_step_score, beam_size)    # nb_samples, beam_size
    top_score = K.log(top_score)

    prev_output_index = K.cast(K.zeros_like(top_score), dtype=K.dtype(top_indice))    # nb_samples, beam_size
    output_label_id = top_indice    # nb_samples, beam_size
    current_input = embedding.call(K.reshape(output_label_id, shape=(-1,)))    # nb_samples* beam_siz, input_dim

    current_state = repeat(_state, beam_size)    # shape: nb_samples*beam_size, state_dim
    current_score = K.reshape(top_score, shape=(-1,))    # shape: nb_samples*beam_size

    return top_score, output_label_id, prev_output_index, current_input, current_state, current_score

def beam_search_one_step(step_score, state, current_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding, tensors_to_debug=None):
    output_dim = K.shape(step_score)[1]    # nb_samples*beam_size, output_dim
    # accumulate accumulated_score
    accumulated_score = K.expand_dims(current_score) + K.log(step_score)    # nb_samples*beam_size, output_dim
    # select top output labels for each sample
    accumulated_score = K.reshape(accumulated_score, shape=pack([number_of_samples, beam_size * output_dim ]))    # nb_samples, beam_size* output_dim
    top_score , top_indice = top_k (accumulated_score, beam_size)    # -1, beam_size
    # update accumulated output accumulated_score
    output_score_list.append (top_score)
    current_score = K.reshape(top_score, shape=(-1,))    # nb_samples * beam_size

    # update output label and previous output index
    # top_indice = beam_id * output_dim + output_label_id
    prev_output_index = top_indice // output_dim
    prev_output_index_list.append(prev_output_index)
    output_label_id = top_indice - prev_output_index * output_dim
    output_label_id_list.append (output_label_id)
    # update current input and current_state
    current_input = embedding.call(K.reshape(output_label_id, shape=(-1,)))    # nb_samples* beam_siz, input_dim
    # state : nb_samples*beam_size, state_dim
    # first reshape state to nb_samples, beam_size, state_dim
    # then gather by sample to get a tensor with the shape: nb_samples, beam_size, state_dim
    # finally reshape to nb_samples*beam_size, state_dim
    # note that prev_output_index has a shape of -1, beam_size, so should be reshape to nb_samples, beam_size before calling gather_by_sample
    current_state = K.reshape (gather_by_sample(K.reshape(state, shape=pack([number_of_samples , beam_size , state_dim ])), K.reshape(prev_output_index, shape=pack([number_of_samples, beam_size]))), shape=pack([number_of_samples * beam_size , state_dim ]))
    if tensors_to_debug is not None:
        tensors_to_debug += [accumulated_score, top_score, top_indice]
    return current_score, current_input, current_state

# output, current_state = self.step(current_input, current_state, context)
def beam_search(initial_input, initial_state, context, context_mask, embedding, step_func, beam_size=1, max_length=20):
    '''Returns a lattice with time steps = max_length and beam size = beam_size; each node of the lattice at time step t has a parent node at time step t-1, an accumulated score, and a label as its output.

    # Parameters
    ----------
    initial_input : a tensor with a shape of nb_samples, representing the initial input used by the step function
    initial_state: a tensor with a shape of nb_samples,state_dim, representing the initial state used by the step function
    context: a float tensor with a shape of nb_samples,time_steps, context_dim, representing the context tensor used by the step function
    context_mask: a boolean tensor with a shape of nb_samples, time_steps, representing the  mask of the context tensor
    embedding: an embedding layer that maps input/output labels to their embedding
    step_func: in a form like step_func(current_input, current_state, context), which returns a score tensor and a tensor representing the updated state
    beam_size: beam size
    max_length: max time steps to expand

    # Returns
    ------
    output_label_id_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type int32, representing labels of nodes
    prev_output_index_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type int32, representing parent's indexes (in the range of 0..beam_size-1) of nodes
    output_score_tensor: a tensor with a shape of max_length, nb_samples, beam_size of type float32, representing accumulated scores of nodes
    '''
    # run first step
    step_score, state = step_func(initial_input, initial_state, context, context_mask)    # nb_samples , output_dim
    top_score, output_label_id, prev_output_index, current_input, current_state, current_score = beam_search_first_step(step_score, state, embedding, beam_size)

    number_of_samples = K.shape(initial_input)[0]
    state_dim = K.shape(initial_state)[K.ndim(initial_state) - 1]

    output_score_list = []    # nb_samples, beam_size
    output_label_id_list = []
    prev_output_index_list = []    # the index of candidate from which current label id is generated

    output_score_list.append(top_score)
    prev_output_index_list.append(prev_output_index)
    output_label_id_list.append(output_label_id)

    context = repeat(context, beam_size)    # shape: nb_samples*beam_size, time_steps, context_input_dim
    if context_mask is not None:
        context_mask = repeat(context_mask, beam_size)    # shape: nb_samples*beam_size, time_steps

    for _ in xrange(max_length - 1):
        step_score, state = step_func(current_input, current_state, context, context_mask)    # nb_samples*beam_size , output_dim
        current_score, current_input, current_state = beam_search_one_step(step_score, state, current_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding)
    # returning a list instead of a tuple of tensors so that keras will know multiple output tensors are generated
    return [pack(output_label_id_list), pack(prev_output_index_list), pack(output_score_list)]

# TODO: fix the issue that short candidates will be filtered
def get_k_best_from_lattice(lattice, k=1, eos_id=None, tensors_to_debug=None):
    '''Selects top k best path from a lattice in a descending order by their scores

    # Parameters
    ----------
    lattice : a triple consisting of output_label_id_tensor, prev_output_index_tensor and output_score_tensor. This lattice is generated by calling beam_search.
    k: the number of path to select from that lattice
    eos_id: if not None, it is the id of the label that represents the end of sequence

    # Returns
    ------
    sequence: a tensor of type int32 with a shape of nb_samples, k, time_stpes, representing the top-k best sequences
    sequence_score: a tensor of type float32 with a shape of nb_samples, k, representing the scores of the top-k best sequences
    '''
    lattice = [unpack(_) for _ in  lattice]
    for l in lattice: l.reverse()
    output_label_id_list, prev_output_index_list, output_score_list = lattice
    sequence_score, output_indice = top_k (output_score_list[0], k)    # shape: nb_samples,k
    if tensors_to_debug is not None:
        tensors_to_debug.append(sequence_score)
        tensors_to_debug.append(output_indice)

    nb_samples = K.shape(sequence_score)[0]
    # fill sequence and update sequence_score
    sequence = []
    for cur_output_score, output_label_id, prev_output_index in zip(output_score_list, output_label_id_list, prev_output_index_list):
        sequence_score_candidate = K.reshape(gather_by_sample(cur_output_score, output_indice), shape=pack([nb_samples, k]))
        sequence.append (K.reshape(gather_by_sample(output_label_id, output_indice), shape=pack([nb_samples, k])))    # shape: -1,  k, nb_samples could be -1
        if eos_id is not None and len(sequence) > 1:
            cond = K.equal(sequence[-1], eos_id)
            sequence_score = choose_by_cond(cond, sequence_score_candidate, sequence_score)
            if tensors_to_debug is not None:
                tensors_to_debug.append(cond)
                tensors_to_debug.append(sequence_score_candidate)
                tensors_to_debug.append(sequence_score)
        output_indice = gather_by_sample(prev_output_index, output_indice)
        if tensors_to_debug is not None:
            tensors_to_debug.append(output_indice)

    if eos_id is not None and len(sequence) > 1:
        sequence_score, output_indice = top_k(sequence_score, k)
        sequence = [gather_by_sample(_, output_indice) for _ in sequence]

    # reverse the sequence so we get sequence from time step 0, 1, ...,
    sequence.reverse()
    sequence = K.permute_dimensions(pack(sequence), (1, 2, 0))    # time_steps, nb_samples, k -> nb_samples, k, time_steps
    return sequence, sequence_score

def choose_by_cond(cond, _1, _2):
    '''Performs element wise choose from _1 or _2 based on condition cond. At a give position, if the element in cond is 1, select the element from _1 otherwise from _2 from the same position.

    # Parameters
    ----------
    cond : a binary tensor
    _1 : first tensor with the same shape of cond
    _2: second tensor with the shape of cond and the same data type of _1

    # Returns
    ------
    a tensor with the shape of cond and same data type of _1
    '''
    cond = K.cast(cond, _1.dtype)
    return _1 * cond + _2 * (1 - cond)

def pack(tensor_list):
    output = K.pack(tensor_list)
    output.num = len(tensor_list)
    return output

def gather_by_sample_by_val(x, indices):
    ndim = np.ndim(x)
    nb_samples = len(x)
    if ndim >= 2:
        assert len(indices) == nb_samples
        output = [x[i, indices[i]] for i in xrange(nb_samples)]
    else:
        output = x[indices]
    return np.array(output)

def top_k_by_val (score, k, sort=False):
    # select top k along the last dimension
    if sort:
        top_indice = np.argsort(-score)[..., :k]    # nb_samples, k
    else:
        top_indice = np.argpartition(-score, k)[..., :k]    # nb_samples, k

    top_score = gather_by_sample_by_val(score, top_indice)    # nb_samples, k
    # top_indice is np.int (int64)
    return  top_score, top_indice

def beam_search_first_step_by_val(step_score, state, embedding, beam_size):
    '''Defines internal helper function that executes the first step for beam search.

    # Parameters
    ----------
    step_score : a ndarray representing score after executing the first step, which has a shape number_samples, output_dim
    state : a ndarray of a shape of nb_samples, state_dim, representing the state after executing the first step
    embedding : an embedding layer that maps input/output labels to their embedding
    beam_size: beam size

    # Returns
    ------
    top_score: a ndarray with a shape of nb_samples, beam_size, representing scores of top beam_size predictions
    output_label_id: a ndarray with a shape of nb_samples, beam_size, representing the labels of the top beam_size predictions
    prev_output_index: a tensor with a shape of nb_samples, beam_size, with zero as its value
    current_input: a ndarray with a shape of nb_samples*beam_size, input_dim, corresponding to the embedding of output_label_id (reshaped)
    current_state: a ndarray with a shape of nb_samples*beam_size; the beam_size candidates derived from the i^th sample has a state state[i]
    current_score: a ndarray with a shape of nb_samples*beam_size, reshaped version of top_score
    '''
    top_score, top_indice = top_k_by_val(step_score, beam_size)    # nb_samples, beam_size
    top_score = np.log(top_score)

    prev_output_index = np.zeros_like(top_score, dtype=top_indice.dtype)    # nb_samples, beam_size
    output_label_id = top_indice    # nb_samples, beam_size

    current_input = embedding[np.reshape(output_label_id, newshape=(-1,))]    # nb_samples* beam_siz, input_dim
    current_state = np.repeat(state, beam_size, axis=0)    # shape: nb_samples*beam_size, state_dim
    current_score = np.reshape(top_score, newshape=(-1,))    # shape: nb_samples*beam_size

    return top_score, output_label_id, prev_output_index, current_input, current_state, current_score

def beam_search_one_step_by_val(step_score, state, current_output_score, number_of_samples, beam_size, state_dim, output_score_list, prev_output_index_list, output_label_id_list, embedding):
    output_dim = step_score.shape[1]    # nb_samples*beam_size, output_dim
    # accumulate score
    accumulated_score = np.expand_dims(current_output_score, axis=-1) + np.log(step_score)    # nb_samples*beam_size, output_dim
    # select top output labels for each sample
    accumulated_score = np.reshape(accumulated_score, newshape=(number_of_samples, beam_size * output_dim))    # nb_samples, beam_size* output_dim

    top_score, top_indice = top_k_by_val(accumulated_score, beam_size)    # nb_samples, beam_size

    # update accumulated output score
    output_score_list.append (top_score)

    current_output_score = np.reshape(top_score, newshape=(-1,))    # nb_samples * beam_size

    # update output label and previous output index
    # top_indice = beam_id * output_dim + output_label_id
    prev_output_index = top_indice // output_dim
    prev_output_index_list.append(prev_output_index)

    output_label_id = top_indice - prev_output_index * output_dim
    output_label_id_list.append (output_label_id)

    # update current input and current_state
    current_input = embedding[np.reshape(output_label_id, newshape=(-1,))]    # nb_samples* beam_siz, input_dim
    # state : nb_samples*beam_size, state_dim
    # first reshape state to nb_samples, beam_size, state_dim
    # then gather by sample to get a tensor with the shape: nb_samples, beam_size, state_dim
    # finally reshape to nb_samples*beam_size, state_dim
    # note that prev_output_index has a shape of -1, beam_size, so should be reshape to nb_samples, beam_size before calling gather_by_sample
    current_state = gather_by_sample_by_val(np.reshape(state, newshape=(number_of_samples, beam_size, state_dim)), prev_output_index)    # nb_samples, beam_size, state_dim
    current_state = np.reshape(current_state, newshape=(-1, state_dim))    # nb_samples*beam_size
    return current_output_score, current_input, current_state

def beam_search_by_val(initial_input, initial_state, context, context_mask, embedding, step_func, beam_size=1, max_length=20):
    '''Returns a lattice with time steps = max_length and beam size = beam_size; each node of the lattice at time step t has a parent node at time step t-1, an accumulated score, and a label as its output.

    # Parameters
    ----------
    initial_input : a ndarray with a shape of nb_samples, representing the initial input used by the step function
    initial_state: a ndarray  with a shape of nb_samples,state_dim, representing the initial state used by the step function
    context: a ndarray with a shape of nb_samples,time_steps, context_dim, representing the context tensor used by the step function
    context_mask: a boolean ndarray with a shape of nb_samples, time_steps, representing the  mask of the context tensor
    embedding: an embedding ndarray that maps input/output labels to their embedding
    step_func: in a form like step_func(current_input, current_state, context), which returns a score tensor and a tensor representing the updated state
    beam_size: beam size
    max_length: max time steps to expand

    # Returns
    ------
    output_label_id: a ndarray with a shape of max_length, nb_samples, beam_size of type int32, representing labels of nodes
    prev_output_index_tensor: a ndarray with a shape of max_length, nb_samples, beam_size of type int32, representing parent's indexes (in the range of 0..beam_size-1) of nodes
    output_score_tensor: a ndarray with a shape of max_length, nb_samples, beam_size of type float32, representing accumulated scores of nodes
    '''
    # run first step
    # state: states of all outputs, from which we will select top beam_size
    step_score, state = step_func(initial_input, initial_state, context, context_mask)    # nb_samples , output_dim
    # cur_state: states of top candidates
    top_score, output_label_id, prev_output_index, current_input, current_state, current_score = beam_search_first_step_by_val(step_score,
                                                                                                                               state,
                                                                                                                               embedding,
                                                                                                                               beam_size)

    number_of_samples = len(initial_input)
    state_dim = initial_state.shape[-1]

    output_score_list = []    # nb_samples, beam_size
    output_label_id_list = []
    prev_output_index_list = []    # the index of up layer candidate from which current label id is generated

    output_score_list.append(top_score)
    prev_output_index_list.append(prev_output_index)
    output_label_id_list.append(output_label_id)

    context = np.repeat(context, beam_size, axis=0)    # shape: nb_samples*beam_size, time_steps, context_input_dim
    if context_mask is not None:
        context_mask = np.repeat(context_mask, beam_size, axis=0)    # shape: nb_samples*beam_size, time_steps

    for _ in xrange(max_length - 1):
        step_score, state = step_func(current_input, current_state, context, context_mask)    # nb_samples*beam_size , output_dim
        current_score, current_input, current_state = beam_search_one_step_by_val(step_score,
                                                                                  state,
                                                                                  current_score,
                                                                                  number_of_samples,
                                                                                  beam_size,
                                                                                  state_dim,
                                                                                  output_score_list,
                                                                                  prev_output_index_list,
                                                                                  output_label_id_list,
                                                                                  embedding)
    return [output_label_id_list, prev_output_index_list, output_score_list]


class _BestCandidate(object):
    def __init__(self, path, score):
        self.score = score
        self.path = path

    def __cmp__(self, other):    # lower score first
        return cmp(self.score, other.score)

class _BestCandidates(object):
    def __init__(self, maxsize):
        assert maxsize > 0
        self.candidates = []
        self.maxsize = maxsize

    def is_full(self):
        return len(self.candidates) >= self.maxsize

    def get_minimal_score(self):
        if self.is_full():
            return self.candidates[0].score
        else:
            return float('-Inf')

    def put(self, candidate):
        if not self.is_full():
            self.candidates.append(candidate)
            if self.is_full():
                self.candidates.sort()
            return True
        else:
            worse_best = self.candidates[0]
            if worse_best.score >= candidate.score:
                return False
            # replace the worse best candidate
            self.candidates.remove(worse_best)
            import bisect

            pos = bisect.bisect_left(self.candidates, candidate)
            self.candidates.insert(pos, candidate)
            return True

    def get_candidates(self):
        candidates = [best for best in self.candidates]
        candidates.sort(reverse=True)
        return candidates

def get_k_best_from_lattice_by_val(lattice, k=1, eos_id=None):
    '''Selects top k best path from a lattice in a descending order by their scores

    # Parameters
    ----------
    lattice : a list consisting of the output_label_id_ndarray, prev_output_index_tensor and output_score_ndarray. 
    This lattice is generated by beam_search, each with the shape of nb_samples,beam_size
    k: the number of path to select from that lattice
    eos_id: if not None, it is the id of the label that represents the end of sequence

    # Returns
    ------
    sequence: a ndarray of type int32 with a shape of nb_samples, k, time_stpes, representing the top-k best sequences
    sequence_score: a ndarray of type float32 with a shape of nb_samples, k, representing the scores of the top-k best sequences
    '''
    # process from left to right, for each sample, on each step, we perform the following:
    # 1. If the sample ready early stopped, ignore the the processing of this sample; otherwise go to 2
    # 2. check each candidate if it still live:
    #    2.1.if its parent is dead, it is dead;
    #    2.2 if its parent is live, and its parent ends of EOS, it is live, and mark its parent is dead, and copy score and path from its parent
    # 3. for each live candidate
    #    2.1.if that candidate is end of EOS, mark that candidate as dead, and try to add this candidate to the best queue of this sample; else do 3.2
    #    2.2 if this candidate has a score lower than the lowest score of current best candidates of that sample, mark this candidate to dead
    # 4. if no live candidates, mark early stop for this sample
    # 5. after all steps processed, finally for any non-early stop sample, add the their live partial candidates decently ordered by their score until best list is full
    output_label_id_list, prev_output_index_list, output_score_list = lattice

    nb_samples = len(output_label_id_list[0])
    nb_candidates = len(output_label_id_list[0][0])

    path_cur = [[[] for __ in xrange(nb_candidates)] for _  in xrange(nb_samples)]
    live_cur = [[True for __ in xrange(nb_candidates)] for _ in xrange(nb_samples)]

    # record the path and score of each best
    k_best = [_BestCandidates(maxsize=k) for _ in xrange(nb_samples)]
    k_best_early_stop = [False for _ in xrange(nb_samples)]
    # process the first
    path_prev = [[[output_label_id_list[0][i][j]] for j in xrange(nb_candidates)] for i in xrange(nb_samples)]

    live_prev = [[True for __ in xrange(nb_candidates)] for _ in xrange(nb_samples)]

    for i in xrange(nb_samples):
        sample_output_label = output_label_id_list[0][i]
        sample_live_prev = live_prev [i]
        sample_k_best = k_best[i]
        sample_score_cur = output_score_list[0][i]
        sample_ordered_indice = np.argsort(-sample_score_cur)

        for j in sample_ordered_indice:
            if sample_output_label[j] == eos_id:
                sample_live_prev[j] = False
                score = sample_score_cur[j]
                path = [sample_output_label[j]]
                best_candidate = _BestCandidate(path=path, score=score)
                sample_k_best.put(best_candidate)
            else:
                if sample_k_best.get_minimal_score() >= sample_score_cur[j]:
                    sample_live_prev[j] = False
                else:
                    sample_live_prev[j] = True

        k_best_early_stop[i] = not any(sample_live_prev)

    score_prev = np.array([[output_score_list[0][i][j] for j in xrange(nb_candidates)] for i in xrange(nb_samples)])
    score_cur = np.array([[0. for __ in xrange(nb_candidates)] for _ in xrange(nb_samples)])

    for cur_output_score, output_label_id, prev_output_index in zip(output_score_list[1:], output_label_id_list[1:], prev_output_index_list[1:]):
        for i in xrange(nb_samples):
            if k_best_early_stop[i]:
                continue

            sample_k_best = k_best[i]
            sample_score = cur_output_score[i]
            sample_score_prev = score_prev[i]
            sample_score_cur = score_cur[i]

            sample_live = live_cur[i]
            sample_live_prev = live_prev [i]
            sample_output_label = output_label_id[i]
            sample_prev_output_index = prev_output_index[i]
            sample_path_prev = path_prev[i]
            sample_path = path_cur[i]

            # update sample_live, sample_cur_score, and path
            for j in xrange(nb_candidates):
                parent_index = sample_prev_output_index[j]
                if not sample_live_prev[parent_index]:
                    sample_live[j] = False
                    continue
                # parent is live, so it is live
                sample_live[j] = True

                is_parent_end_of_eos = sample_path_prev[parent_index][-1] == eos_id
                if is_parent_end_of_eos:
                    sample_live_prev[parent_index] = False    # to prevent other children from this parent being live
                    sample_path[j] = sample_path_prev[parent_index]
                    sample_score_cur[j] = sample_score_prev[parent_index]
                else:
                    sample_path[j] = sample_path_prev[parent_index] + [sample_output_label[j]]
                    sample_score_cur[j] = sample_score[j]

            # check any live candidates end of eos, and update score_prev path_prev, live_pre for live candidates
            ordered_indice = np.argsort(-sample_score_cur)

            for j in ordered_indice:
                sample_live_prev[j] = sample_live[j]
                if not sample_live[j]:
                    continue
                # current candidate is live and end of eos, we add it to best and mark it as un-live
                is_cur_end_of_ens = sample_path[j][-1] == eos_id
                if is_cur_end_of_ens:
                    sample_live_prev[j] = False
                    score = sample_score_cur[j]
                    path = sample_path[j]
                    best_candidate = _BestCandidate(path=path, score=score)
                    sample_k_best.put(best_candidate)
                else:
                    if sample_k_best.get_minimal_score() >= sample_score_cur[j]:
                        sample_live_prev[j] = False
                    else:
                        sample_live_prev[j] = True
                        sample_score_prev[j] = sample_score_cur[j]
                        sample_path_prev[j] = sample_path[j]

            k_best_early_stop[i] = not any(sample_live_prev)

    # last check if not full, and fill with partial live candidate
    for i in xrange(nb_samples):
        if k_best_early_stop[i]:
            continue
        sample_k_best = k_best[i]
        if sample_k_best.is_full():
            continue
        sample_score_prev = score_prev[i]
        sample_path_prev = path_prev[i]
        sample_live_prev = live_prev [i]
        ordered_indice = np.argsort(-sample_score_prev)
        for j in ordered_indice:
            if sample_live_prev[j]:
                best_candidate = _BestCandidate(path=sample_path_prev[j], score=sample_score_prev[j])
                sample_k_best.put(best_candidate)
                if sample_k_best.is_full():
                    break

    candidates = [sample_k_best.get_candidates() for sample_k_best in k_best]
    sequence = [[sample_candidate.path for sample_candidate  in sample_candidates] for sample_candidates in candidates]
    sequence_score = [[sample_candidate.score for sample_candidate  in sample_candidates] for sample_candidates in candidates]
    return np.array(sequence), np.array(sequence_score)

if K._BACKEND == 'theano':
    import theano
    from theano import tensor as T

    def unpack(x, num=None):
        '''Gets a list of tensors by slicing a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : number of tensors to return

        # Returns
        ------
        a list of tensors sliced by the first dimension of the input tensor
        '''
        if num is None:
            assert hasattr(x, 'num')
            num = x.num
        return [x[i] for i in range(num)]

    def top_k(x, k=1):
        """Finds values and indices of the `k` largest entries for the last dimension sorted by value in descent.

        If the input is a vector (rank-1), finds the `k` largest entries in the vector
        and outputs their values and indices as vectors.  Thus `values[j]` is the
        `j`-th largest entry in `input`, and its index is `indices[j]`.

        For matrices (resp. higher rank input), computes the top `k` entries in each
        row (resp. vector along the last dimension).  Thus,

            values.shape = indices.shape = input.shape[:-1] + [k]

        If two elements are equal, the lower-index element appears first.

        # Parameters
        ----------
        input: 1-D or higher `Tensor` with last dimension at least `k`.
        k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices).

        # Returns:
        ----------
        values: The `k` largest elements along each last dimensional slice.
        indices: The indices of `values` within the last dimension of `input`.
        """
        x_sorted = T.sort(x)
        x_sort_arg = T.argsort(x)
        ndim = x.ndim
        if ndim == 1:
            x_sorted = x_sorted[-k:]
            x_sorted = x_sorted[::-1]
            x_sort_arg = x_sort_arg[-k:]
            x_sort_arg = x_sort_arg[::-1]
            return x_sorted, x_sort_arg
        else:
            new_shape = T.stack(*([x.shape[i] for i in range(ndim - 1)] + [k]))
            x_sorted = T.reshape(x_sorted, newshape=(-1, x.shape[-1]))[:, -k:]
            x_sorted = x_sorted[:, ::-1]
            x_sorted = T.reshape(x_sorted, new_shape, ndim=ndim)
            x_sort_arg = T.reshape(x_sort_arg, newshape=(-1, x.shape[-1]))[:, -k:]
            x_sort_arg = x_sort_arg[:, ::-1]
            x_sort_arg = T.reshape(x_sort_arg, new_shape, ndim=ndim)
            return x_sorted, x_sort_arg

    def gather_by_sample(x, indices):
        '''Performs gather operation along the first dimension, i.e., ret[i] = gather( x[i], indices[i]).
        For example, when x is a matrix, and indices is a vector, it selects one element for each row from x.
        Note that this is different from gather, which selects |indices| ndim-1 sub tensors (i.e., x[i], where i = indices[:::]) from x

        # Parameters
        ----------
        x : a tensor with a shape nb_samples, ...; its number of dimensions >= 2
        indices : a tensor of type int with a shape nb_sample,...; its number of dimensions <= # of dimensions of x - 1

        # Returns
        ------
        a tensor with the shape of nb_samples, ..., where ret[i,:::,:::]= x[i,indices[i,:::],:::]; and its number of dimensions = # dimensions of x + # dimension of indices - 2
        '''
        x_shape = K.shape(x)
        nb_samples = x_shape[0]
        seq = T.arange(nb_samples)
        def _step(i, x, indices):
            x_i = K.gather(x, i)
            indices_i = K.gather(indices, i)
            return K.gather(x_i, indices_i)
        outputs, _ = theano.scan(fn=_step, sequences=seq, non_sequences=[x, indices], outputs_info=None)
        return outputs

    def dot(x, y):
        return T.dot(x, y)

    def clip_norm(g, c, n):
        if c > 0:
            g = K.switch(n >= c, g * c / n, g)
        return g

    def rnn(step_function, inputs, initial_states,
            go_backwards=False, mask=None, constants=None,
            unroll=False, input_length=None, output_dim=None):
        '''Iterates over the time dimension of a tensor.

        # Arguments
            inputs: tensor of temporal data of shape (samples, time, ...)
                (at least 3D).
            step_function:
                Parameters:
                    input: tensor with shape (samples, ...) (no time dimension),
                        representing input for the batch of samples at a certain
                        time step.
                    states: list of tensors.
                Returns:
                    output: tensor with shape (samples, ...) (no time dimension),
                    new_states: list of tensors, same length and shapes
                        as 'states'.
            initial_states: tensor with shape (samples, ...) (no time dimension),
                containing the initial values for the states used in
                the step function.
            go_backwards: boolean. If True, do the iteration over
                the time dimension in reverse order.
            mask: binary tensor with shape (samples, time),
                with a zero for every element that is masked.
            constants: a list of constant values passed at each step.
            unroll: whether to unroll the RNN or to use a symbolic loop (`scan`).
            input_length: must be specified if using `unroll`.
            output_dim: not used.

        # Returns
            A tuple (last_output, outputs, new_states).
                last_output: the latest output of the rnn, of shape (samples, ...)
                outputs: tensor with shape (samples, time, ...) where each
                    entry outputs[s, t] is the output of the step function
                    at time t for sample s.
                new_states: list of tensors, latest states returned by
                    the step function, of shape (samples, ...).
        '''
        ndim = inputs.ndim
        assert ndim >= 3, 'Input should be at least 3D.'

        if unroll:
            if input_length is None:
                raise Exception('When specifying `unroll=True`, an `input_length` '
                                'must be provided to `rnn`.')

        axes = [1, 0] + list(range(2, ndim))
        inputs = inputs.dimshuffle(axes)

        if constants is None:
            constants = []

        if mask is not None:
            if mask.ndim == ndim - 1:
                mask = K.expand_dims(mask)
            assert mask.ndim == ndim
            mask = mask.dimshuffle(axes)

            if unroll:
                indices = list(range(input_length))
                if go_backwards:
                    indices = indices[::-1]

                successive_outputs = []
                successive_states = []
                states = initial_states
                for i in indices:
                    output, new_states = step_function(inputs[i], states + constants)

                    if len(successive_outputs) == 0:
                        prev_output = K.zeros_like(output)
                    else:
                        prev_output = successive_outputs[-1]

                    output = T.switch(mask[i], output, prev_output)
                    kept_states = []
                    for state, new_state in zip(states, new_states):
                        kept_states.append(T.switch(mask[i], new_state, state))
                    states = kept_states

                    successive_outputs.append(output)
                    successive_states.append(states)

                outputs = T.stack(*successive_outputs)
                states = []
                for i in range(len(successive_states[-1])):
                    states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
            else:
                # build an all-zero tensor of shape (samples, output_dim)
                initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
                # Theano gets confused by broadcasting patterns in the scan op
                initial_output = T.unbroadcast(initial_output, 0, 1)

                def _step(input, mask, output_tm1, *states):
                    output, new_states = step_function(input, states)
                    # output previous output if masked.
                    output = T.switch(mask, output, output_tm1)
                    return_states = []
                    for state, new_state in zip(states, new_states):
                        return_states.append(T.switch(mask, new_state, state))
                    return [output] + return_states

                results, _ = theano.scan(
                    _step,
                    sequences=[inputs, mask],
                    outputs_info=[initial_output] + initial_states,
                    non_sequences=constants,
                    go_backwards=go_backwards)

                # deal with Theano API inconsistency
                if type(results) is list:
                    outputs = results[0]
                    states = results[1:]
                else:
                    outputs = results
                    states = []
        else:
            if unroll:
                indices = list(range(input_length))
                if go_backwards:
                    indices = indices[::-1]

                successive_outputs = []
                successive_states = []
                states = initial_states
                for i in indices:
                    output, states = step_function(inputs[i], states + constants)
                    successive_outputs.append(output)
                    successive_states.append(states)
                outputs = T.stack(*successive_outputs)
                states = []
                for i in range(len(successive_states[-1])):
                    states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

            else:
                def _step(input, *states):
                    output, new_states = step_function(input, states)
                    return [output] + new_states

                results, _ = theano.scan(
                    _step,
                    sequences=inputs,
                    outputs_info=[None] + initial_states,
                    non_sequences=constants,
                    go_backwards=go_backwards)

                # deal with Theano API inconsistency
                if type(results) is list:
                    outputs = results[0]
                    states = results[1:]
                else:
                    outputs = results
                    states = []

        outputs = T.squeeze(outputs)
        last_output = outputs[-1]

        axes = [1, 0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        states = [T.squeeze(state[-1]) for state in states]
        return last_output, outputs, states

    def shift_right(x):
        '''Gets one right shifted along time dimension of x, padding with zeros
        # Parameters
        ----------
        x : a tensor of shape nb_samples, time_steps, input_dim
 
        # Returns
        ------
        One right shifted tensor
        '''
        y = K.zeros_like(x)
        return T.set_subtensor(y[:, 1:, :], x[:, :-1, :])

    def foreach(x, step_func, dtype=None, name=None):
        '''Process each element in x and returns all the processed outputs in a tensor.
        # Parameters
        ----------
        x : a tensor
        step_func: a function that process an element of the input tensor and output a new tensor, e.g., lambda xi: xi+2.
        dtype: dtype of the output tensor. By default output tensor has the same dtype as x
        # Returns
        ------
        A tensor that packs all the outputs.
        '''
        return theano.scan(fn=step_func, sequences=[x], name=name)[0]

    def scan(fn, sequences, outputs_initials, name=None):
        '''Process multiple sequences, and returns a list of tensors. Each output tensor list corresponds to one tensor in the outputs_initials.
        # Parameters
        ----------
        sequences : a list of tensors
        fn: a function that process previous output tensors and current input tensors, and returns current output tensors
        outputs_initials: initial output tensors
        name: name of the returned tensor
        # Returns
        ------
        A list of output tensors.
        '''
        # warning: updates dictionary ignored
        return theano.scan(fn, sequences=sequences, outputs_info=outputs_initials, name=name)[0]

    def random_multinomial(n=1, pvals=None, dtype=_FLOATX, seed=None):
        if seed is None:
            seed = np.random.randint(1, 10e6)
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=n, pvals=pvals, dtype=dtype)

elif K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def unpack(x, num=None):
        '''Gets a list of tensors by slicing a tensor along its first dimension.

        # Parameters
        ----------
        x : a tensor whose dimensions >= 1
        num : number of tensors to return; if None, x's shape of its first dimension must be specified

        # Returns
        ------
        a list of tensors sliced by the first dimension of the input tensor
        '''
        return tf.unpack(x, num=num)



    def top_k(x, k=1, sorted_by_value_descent=True):
        """Finds values and indices of the `k` largest entries for the last dimension.

        If the input is a vector (rank-1), finds the `k` largest entries in the vector
        and outputs their values and indices as vectors.  Thus `values[j]` is the
        `j`-th largest entry in `input`, and its index is `indices[j]`.

        For matrices (resp. higher rank input), computes the top `k` entries in each
        row (resp. vector along the last dimension).  Thus,

            values.shape = indices.shape = input.shape[:-1] + [k]

        If two elements are equal, the lower-index element appears first.

        # Parameters
        ----------
        input: 1-D or higher `Tensor` with last dimension at least `k`.
        k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices).
        sorted_by_value_descent: If true the resulting `k` elements will be sorted_by_value_descent by the values in descending order.

        # Returns:
        ----------
        values: The `k` largest elements along each last dimensional slice.
        indices: The indices of `values` within the last dimension of `input`.
        """
        return tf.nn.top_k(x, k)

    def gather_by_sample(x, indices):
        '''Performs gather operation along the first dimension, i.e., ret[i] = gather( x[i], indices[i]).
        For example, when x is a matrix, and indices is a vector, it selects one element for each row from x.
        Note that this is different from gather, which selects |indices| ndim-1 sub tensors (i.e., x[i], where i = indices[:::]) from x

        # Parameters
        ----------
        x : a tensor with a shape nb_samples, ...; its number of dimensions >= 2
        indices : a tensor of type int with a shape nb_sample,...; its number of dimensions <= # of dimensions of x - 1

        # Returns
        ------
        a tensor with the shape of nb_samples, ..., where ret[i,:::,:::]= x[i,indices[i,:::],:::]; and its number of dimensions = # dimensions of x + # dimension of indices - 2
        '''
        x_shape = K.shape(x)
        nb_samples = x_shape[0]
        ones = tf.ones(shape=pack([nb_samples]), dtype='int32')
        elems = tf.scan(lambda prev, one: prev + one , ones, initializer=tf.constant(-1, dtype='int32'))
        def _step(prev, i):
            x_i = K.gather(x, i)
            indices_i = K.gather(indices, i)
            return K.gather(x_i, indices_i)
        return tf.scan(_step , elems, initializer=tf.zeros(shape=x_shape[1:], dtype=x.dtype))

    # support None
    def dot(x, y):
        '''Multiplies 2 tensors.
        When attempting to multiply a ND tensor
        with a ND tensor, reproduces the Theano behavior
        (e.g. (2, 3).(4, 3, 5) = (2, 4, 5))
        '''
        ndim_x = K.ndim(x)
        ndim_y = K.ndim(y)

        if ndim_x is not None and ndim_x > 2 or ndim_y > 2:
            x_shape = tf.shape(x)
            y_shape = tf.shape(y)
            y_permute_dim = list(range(ndim_y))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, pack([-1, x_shape[ndim_x - 1]]))
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), pack([y_shape[ndim_y - 2], -1]))
            target_shape = [x_shape[i] for i in range(ndim_x - 1)] + [y_shape[i] for i in range(ndim_y - 2)] + [y_shape[ndim_y - 1]]
            return tf.reshape(tf.matmul(xt, yt), pack(target_shape))
        out = tf.matmul(x, y)
        return out

    def clip_norm(g, c, n):
        if c > 0:
            f = tf.python.control_flow_ops.cond(tf.cast(n >= c, 'bool'),
                                        lambda: c / n,
                                        lambda: tf.constant(1.0))
        return tf.scalar_mul(f, g)

    # Mainly copied from Keras, with one small update to support the case that the output_dim of the rnn_cell is not the same as the final output_dim
    def rnn(step_function, inputs, initial_states,
            go_backwards=False, mask=None, constants=None,
            unroll=False, input_length=None, output_dim=None):
        '''Iterates over the time dimension of a tensor.

        # Arguments
            inputs: tensor of temporal data of shape (samples, time, ...)
                (at least 3D).
            step_function:
                Parameters:
                    input: tensor with shape (samples, ...) (no time dimension),
                        representing input for the batch of samples at a certain
                        time step.
                    states: list of tensors.
                Returns:
                    output: tensor with shape (samples, output_dim) (no time dimension),
                    new_states: list of tensors, same length and shapes
                        as 'states'. The first state in the list must be the
                        output tensor at the previous timestep.
            initial_states: tensor with shape (samples, output_dim) (no time dimension),
                containing the initial values for the states used in
                the step function.
            go_backwards: boolean. If True, do the iteration over
                the time dimension in reverse order.
            mask: binary tensor with shape (samples, time, 1),
                with a zero for every element that is masked.
            constants: a list of constant values passed at each step.
            unroll: with TensorFlow the RNN is always unrolled, but with Theano you
                can use this boolean flag to unroll the RNN.
            input_length: not relevant in the TensorFlow implementation.
                Must be specified if using unrolling with Theano.
            output_dim: the output dim of the output of the step function. If not set, it is set to the dim of the first state tensor.

        # Returns
            A tuple (last_output, outputs, new_states).

            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
        '''
        ndim = len(inputs.get_shape())
        assert ndim >= 3, 'Input should be at least 3D.'
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))

        if constants is None:
            constants = []

        if unroll:
            if not inputs.get_shape()[0]:
                raise Exception('Unrolling requires a fixed number of timesteps.')

            states = initial_states
            successive_states = []
            successive_outputs = []

            input_list = tf.unpack(inputs)
            if go_backwards:
                input_list.reverse()

            if mask is not None:
                # Transpose not supported by bool tensor types, hence round-trip to uint8.
                mask = tf.cast(mask, tf.uint8)
                if len(mask.get_shape()) == ndim - 1:
                    mask = K.expand_dims(mask)
                mask = tf.cast(tf.transpose(mask, axes), tf.bool)
                mask_list = tf.unpack(mask)

                if go_backwards:
                    mask_list.reverse()

                for input, mask_t in zip(input_list, mask_list):
                    output, new_states = step_function(input, states + constants)

                    # tf.select needs its condition tensor to be the same shape as its two
                    # result tensors, but in our case the condition (mask) tensor is
                    # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
                    # broadcast the mask to match the shape of A and B. That's what the
                    # tile call does, is just repeat the mask along its second dimension
                    # ndimensions times.
                    tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(output)[1]]))

                    if len(successive_outputs) == 0:
                        prev_output = K.zeros_like(output)
                    else:
                        prev_output = successive_outputs[-1]

                    output = tf.select(tiled_mask_t, output, prev_output)

                    return_states = []
                    for state, new_state in zip(states, new_states):
                        # (see earlier comment for tile explanation)
                        tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(new_state)[1]]))
                        return_states.append(tf.select(tiled_mask_t, new_state, state))

                    states = return_states
                    successive_outputs.append(output)
                    successive_states.append(states)
                    last_output = successive_outputs[-1]
                    new_states = successive_states[-1]
                    outputs = tf.pack(successive_outputs)
            else:
                for input in input_list:
                    output, states = step_function(input, states + constants)
                    successive_outputs.append(output)
                    successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.pack(successive_outputs)

        else:
            from tensorflow.python.ops.rnn import _dynamic_rnn_loop

            if go_backwards:
                inputs = tf.reverse(inputs, [True] + [False] * (ndim - 1))

            states = initial_states
            nb_states = len(states)
            if nb_states == 0:
                raise Exception('No initial states provided.')
            elif nb_states == 1:
                state = states[0]
            else:
                state = tf.concat(1, states)

            state_size = int(states[0].get_shape()[-1])

            if mask is not None:
                if go_backwards:
                    mask = tf.reverse(mask, [True] + [False] * (ndim - 1))

                # Transpose not supported by bool tensor types, hence round-trip to uint8.
                mask = tf.cast(mask, tf.uint8)
                if len(mask.get_shape()) == ndim - 1:
                    mask = K.expand_dims(mask)
                mask = tf.transpose(mask, axes)
                inputs = tf.concat(2, [tf.cast(mask, inputs.dtype), inputs])

                def _step(input, state):
                    if nb_states > 1:
                        states = []
                        for i in range(nb_states):
                            states.append(state[:, i * state_size: (i + 1) * state_size])
                    else:
                        states = [state]
                    mask_t = tf.cast(input[:, 0], tf.bool)
                    input = input[:, 1:]
                    output, new_states = step_function(input, states + constants)
                    # output zero tensor if it is masked as zero
                    output = tf.select(mask_t, output, K.zeros_like(output))
                    new_states = [tf.select(mask_t, new_states[i], states[i]) for i in range(len(states))]

                    if len(new_states) == 1:
                        new_state = new_states[0]
                    else:
                        new_state = tf.concat(1, new_states)

                    return output, new_state
            else:
                def _step(input, state):
                    if nb_states > 1:
                        states = []
                        for i in range(nb_states):
                            states.append(state[:, i * state_size: (i + 1) * state_size])
                    else:
                        states = [state]
                    output, new_states = step_function(input, states + constants)

                    if len(new_states) == 1:
                        new_state = new_states[0]
                    else:
                        new_state = tf.concat(1, new_states)
                    return output, new_state

            _step.state_size = state_size * nb_states
            if output_dim is None:
                _step.output_size = state_size
            else:
                _step.output_size = output_dim

            (outputs, final_state) = _dynamic_rnn_loop(
                _step,
                inputs,
                state,
                parallel_iterations=32,
                swap_memory=True,
                sequence_length=None)

            if nb_states > 1:
                new_states = []
                for i in range(nb_states):
                    new_states.append(final_state[:, i * state_size: (i + 1) * state_size])
            else:
                new_states = [final_state]

            # all this circus is to recover the last vector in the sequence.
            begin = tf.pack([tf.shape(outputs)[0] - 1] + [0] * (ndim - 1))
            size = tf.pack([1] + [-1] * (ndim - 1))
            last_output = tf.slice(outputs, begin, size)
            last_output = tf.squeeze(last_output, [0])

        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        return last_output, outputs, new_states

    def shift_right(x):
        '''Gets one right shifted along time dimension of x, padding with zeros
        # Parameters
        ----------
        x : a tensor of shape nb_samples, time_steps, input_dim

        # Returns
        ------
        One right shifted tensor
        '''
        last_removed = K.reverse(K.reverse(x, axes=1)[:, 1:, :], axes=1)
        padding = K.expand_dims(K.zeros_like(x[:, 0, :]), dim=1)
        return K.concatenate([padding, last_removed], axis=1)


    def foreach(x, step_func, dtype=None, name=None):
        '''Process each element in x and returns all the processed outputs in a tensor. 
        # Parameters
        ----------
        x : a tensor
        step_func: a function that process an element of the input tensor and output a new tensor, e.g., lambda xi: xi+2.
        dtype: dtype of the output tensor. By default output tensor has the same dtype as x
        # Returns
        ------
        A tensor that packs all the outputs.
        '''
        from tensorflow.python.ops import tensor_array_ops
        size = K.shape(x)[0]
        accs_ta = tensor_array_ops.TensorArray(dtype=dtype if dtype else x.dtype,
                                                  size=size,
                                                  dynamic_size=False,
                                                  infer_shape=True)
        i = tf.constant(0)
        def b(i, tas):
            output = step_func(K.gather(x, i))
            tas = tas.write(i, output)
            return (i + 1, tas)

        _1, outputs = tf.while_loop(lambda i, _: i < size, b, [i, accs_ta])
        return outputs.pack(name=name)

    def scan(fn, sequences, outputs_initials, name=None):
        '''Process multiple sequences, and returns a list of tensors. Each output tensor list corresponds to one tensor in the outputs_initials.
        # Parameters
        ----------
        sequences : a list of tensors
        fn: a function that process previous output tensors and current input tensors, and returns current output tensors
        outputs_initials: initial output tensors
        name: name of the returned tensor
        # Returns
        ------
        A list of output tensors.
        '''
        return tf.scan(fn, elems=sequences, initializer=outputs_initials, name=name)

    def random_multinomial(n=1, pvals=None, dtype=_FLOATX, seed=None):
        samples = tf.multinomial(tf.log(pvals), num_samples=n, seed=seed)
        # one_hot: batch_size, n, nb_classes -> sum: batch_size, nb_classes
        samples = K.sum (tf.one_hot(samples, K.shape(pvals)[K.ndim(pvals) - 1]), axis=-2)
        if dtype and not  dtype == K.dtype(samples):
            samples = K.cast(samples, dtype)
        return samples
