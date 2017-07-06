import tensorflow as tf
import numpy as np

def _flatten_to_2d(tensor):
    print(tensor)
    shape = tf.shape(tensor)
    size  = tf.size(tensor)
    return tf.reshape(tensor, (shape[0], size // shape[0]))

def _total_size(shape):
    size = 0
    for s in shape:
        size += np.prod(s)
    return size

class Memory:
    def __init__(self, count, width, init_state="randomized"):
        self._heads = []
        self._count = count
        self._width = width
        self._init_state = init_state

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
        if self._init_state == "trainable":
            init_values   = np.random.rand(1, self._count, self._width)
            single_memory = tf.Variable(tf.constant(init_values, dtype=dtype))
            memory_state  = tf.tile(single_memory, tf.stack([batch_size, 1, 1]))

        elif self._init_state == "randomized":
            memory_state  = tf.random_normal((batch_size, self._count, self._width), dtype=dtype)

        elif self._init_state == "fixed":
            single_memory = tf.constant(np.random.rand(1, self._count, self._width), dtype=dtype)
            memory_state  = tf.tile(single_memory, tf.stack([batch_size, 1, 1]))

        else:
            memory_state  = tf.tile(self._init_state, tf.stack([batch_size, 1, 1]))

        readouts = [tf.zeros((batch_size, h.output_size), dtype) for h in self.read_heads]
        private  = tuple(h.zero_state(batch_size, dtype) for h in self._heads)

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
        return ro, (ms, nps)

    @property
    def summary_size(self):
        return _total_size(self.summary_shape)

    @property
    def summary_shape(self):
        l = [(self._count, self._width)]
        l += [(h.state_size,) for h in self._heads]
        return tuple(l)

    def pack_summary(self, memory_state):
        flat = list(map(_flatten_to_2d, memory_state[1]))
        return tf.concat([_flatten_to_2d(memory_state[0])]+flat, axis=1)

    def unpack_summary(self, summary):
        batch_size = tf.shape(summary)[0]
        time_size = tf.shape(summary)[1]
        ss = list(map(np.prod, self.summary_shape))
        split = tf.split(summary, ss, axis = 2)
        def _make_shape(shape):
            if isinstance(shape, int):
                return [batch_size, time_size, shape]
            else:
                return [batch_size, time_size]+list(shape)
        return [tf.reshape(data, _make_shape(shape)) for (data, shape) in zip(split, self.summary_shape)]

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

