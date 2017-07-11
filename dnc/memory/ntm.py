import tensorflow as tf
from .memory import Head


# helper function to map to [1, \infty]
def oneplus(x):
    return 1 + tf.nn.softplus(x)

# circular convolution, needed for ntm_weighting; not tested
def circ_conv(w, shifts, shiftws):
    ww = []
    l = w.shape[1].value
    for j in range(len(shifts)):
        i = shifts[j] % l
        ww.append(shiftws[:,j:j+1] * tf.concat([w[:,(l-i):], w[:,:(l-i)]], axis=1))
    return tf.add_n(ww)


def ntm_split_state(control, memory_width, shifts):
    W = memory_width

    # extract raw interface parameters
    raw_strength = control[:, 0]
    raw_gate     = control[:, 1]
    raw_sharping = control[:, 2]
    raw_key      = control[:, 3:3+W]
    raw_shiftws  = control[:, 3+W:3+W+len(shifts)]

    # convert range and broadcast
    strength = oneplus(raw_strength)[:, None]
    gate     = tf.sigmoid(raw_gate)[:,  None]
    sharping = oneplus(raw_sharping)[:, None]
    key      = tf.nn.l2_normalize(raw_key, dim = 1)[:, None ,:]
    shiftws  = tf.nn.softmax(raw_shiftws, dim = 1)
    return strength, gate, sharping, key, shiftws


# algorithm to obtain the weights
# needed: a key, a strength, a gate, a sharpening exponent, and shift weights
# also depends on the previous weights, which are in 'state'
def ntm_weighting(control, memory, state, shifts, idx=0):
    # get the parameters
    strength, gate, sharping, key, shiftws = ntm_split_state(control[:, idx:], memory.shape[2].value, shifts)

    # cosine similarity
    normed_mem = tf.nn.l2_normalize(memory, dim = 2)
    cos_sims   = tf.reduce_sum(key * normed_mem, 2)
    #cos_sims = tf.Print(cos_sims,  [cos_sims])

    # calculate content-, gated-, shifted-, and final-(=sharped) weights
    wc = tf.nn.softmax(strength * cos_sims, dim = 1)
    wg = gate * wc + (1.-gate) * state
    ws = circ_conv(wg, shifts, shiftws)
    w  = tf.pow(ws, sharping)

    # return the normalized weights
    return w / tf.reduce_sum(w, 1)[:,None]


class NTMReadHead(Head):
    def __init__(self, count, width, shifts):
        super(NTMReadHead, self).__init__(
            in_size=width + 3 + len(shifts), 
            out_size=width, 
            state_size=count)
        self._shifts = shifts

    def command(self, control, memory, state):
        weights = ntm_weighting(control, memory, state, self._shifts)
        return weights, weights

    def execute(self, memory, command):
        readout = tf.reduce_sum(memory * command[:,:,None], 1)
        return readout, memory


class NTMWriteHead(Head):
    def __init__(self, count, width, shifts=[-1, 0, 1]):
        super(NTMWriteHead, self).__init__(
            in_size=3 * width + 3 + len(shifts), 
            out_size=None, 
            state_size=count)
        self._shifts = shifts

    def command(self, control, memory, state):
        W = memory.shape[2].value

        # decompose interface parameters
        erase   = tf.sigmoid(control[:, :W])
        write   = control[:, W:2*W]
        weights = ntm_weighting(control, memory, state, self._shifts, idx=2*W)

        return (erase, write, weights), weights

    def execute(self, memory, command):
        erase   = command[0][:, None, :]
        write   = command[1][:, None, :]
        weights = command[2][:, :, None]

        # use * instead of tf.multiply
        memory = memory * (1 - weights * erase)
        memory = memory + weights * write

        return None, memory
