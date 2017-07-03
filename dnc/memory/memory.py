import tensorflow as tf

class Memory:
    def __init__(self, count, width):
        self._heads = []
        self._count = count
        self._width = width

    def add_head(self, head, *args, **kwargs):
        self._heads += [head(count=self._count, width=self._width, *args, **kwargs)]

    @property
    def read_heads(self):
        return [h for h in self._heads if h.output_size is not None]

    @property
    def state_size(self):
        readout_sizes = tuple(h.output_size for h in self.read_heads)
        mem_size = (self._count, self._width)
        head_private = tuple(h.state_size for h in self._heads)
        return readout_sizes, (mem_size, head_private)

    def zero_state(self, batch_size, dtype):
        readouts         = [tf.zeros((batch_size, h.output_size), dtype) for h in self.read_heads]
        memory_state     = tf.random_normal((batch_size, self._count, self._width), dtype=dtype)
        private          = tuple(h.zero_state(batch_size, dtype) for h in self._heads)
        return readouts, (memory_state, private)

    def _prepare_heads(self, controller_vec, memory_state, private_states):
        """ calculates commands of all heads.
        """
        new_private_states = []
        commands = []
        for (i, head) in enumerate(self._heads):
            with tf.variable_scope("head_%i"%i):
                interface_vector = tf.layers.dense(controller_vec, 
                                    head.input_size, 
                                    use_bias = False)

                command, state = head.command(interface_vector,
                                              memory_state,
                                              private_states[i])
                commands += [command]
                new_private_states += [state]

        return tuple(commands), tuple(new_private_states)

    def _execute_heads(self, memory_state, commands):
        readouts = []
        for (head, command) in zip(self._heads, commands):
            readout, memory_state = head.execute(memory_state, command)
            if readout is not None:
                readouts += [readout]
        return readouts, memory_state

    def __call__(self, controller_vec, memory_state):
        cmds, nps = self._prepare_heads(controller_vec, memory_state[0], 
                                        memory_state[1])
        ro, ms    = self._execute_heads(memory_state[0], cmds)
        print("NPS")
        print(nps)
        return ro, (ms, nps)

class SharedState:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

class Head:
    def __init__(self, in_size, out_size, state_size, prefix=""):
        self._in_size = in_size
        self._out_size = out_size
        self._state_size = state_size
        self._prefix = prefix

    def zero_state(self, batch_size, dtype):
        # state contains only the weights of the last time step
        return tf.zeros((batch_size, self.state_size), dtype=dtype)

    def command(self, control, memory, state):
        raise NotImplementedError()

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    @property
    def shared_states(self):
        return self._shared_states

    @property
    def prefix(self):
        return self._prefix

    @property
    def state_size(self):
        # private state
        return self._state_size


#############################################################################


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
