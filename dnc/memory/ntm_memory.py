import tensorflow as tf


# helper function to map to [1, \infty]
def oneplus(x):
    return 1 + tf.log(1 + tf.exp(x))

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



class NTMMemory:
    def __init__(self, width, count):
        self._width = width
        self._count = count

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, self._count, self._width), dtype=dtype)

    @property
    def state_size(self):
        return (self._count, self._width)

    def read_head(self, shifts=[-1, 0, 1]):
        return NTMReadHead(self._count, self._width, shifts)

    def read_heads(self, n, *args, **kwargs):
        # Could be made redundant with parent Memory class
        return [self.read_head(*args, **kwargs) for i in range(n)]

    def write_head(self, shifts=[-1, 0, 1]):
        return NTMWriteHead(self._count, self._width, shifts)

    def write_heads(self, n, *args, **kwargs):
        # Could be made redundant with parent Memory class
        return [self.write_head(*args, **kwargs) for i in range(n)]


class NTMReadHead:
    def __init__(self, count, width, shifts):
        self._in_size = width + 3 + len(shifts)
        self._out_size = width
        self._count = count
        self._width = width
        self._state_size = count
        self._shifts = shifts

    def zero_state(self, batch_size, dtype):
        # state contains only the weights of the last time step
        return tf.zeros((batch_size, self.state_size), dtype=dtype)

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, control, memory, state):
        # decompose interface parameters
        weights = ntm_weighting(control, memory, state, self._shifts)

        # produce weighted readouts
        readout = tf.reduce_sum(memory * weights[:,:,None], 1)
        return readout, weights


class NTMWriteHead:
    def __init__(self, count, width, shifts=[-1, 0, 1]):
        self._in_size = 3 * width + 3 + len(shifts)
        self._count = count
        self._width = width
        self._state_size = count
        self._shifts = shifts

    def zero_state(self, batch_size, dtype):
        # state contains only the weights of the last time step
        return tf.zeros((batch_size, self.state_size), dtype=dtype)

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, control, memory, state):
        W = memory.shape[2].value

        # decompose interface parameters
        erase   = tf.sigmoid(control[:, :W])
        write   = control[:, W:2*W]
        weights = ntm_weighting(control, memory, state, self._shifts, idx=2*W)

        # use this notation instead of tf.expand_dims()
        ex_erase   = erase[:, None, :]
        ex_write   = write[:, None, :]
        ex_weights = weights[:, :, None]

        # use * instead of tf.multiply
        memory = memory * (1 - ex_weights * ex_erase)
        memory = memory + ex_weights * ex_write

        return memory, weights
