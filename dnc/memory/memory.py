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

def _merge_states(list_of_states):
    names = []
    states = []
    for state in list_of_states:
        if state.name not in names:
            states += [state]
            names  += [state.name] 
    return states

class Memory:
    def __init__(self, count, width, init_state="randomized"):
        """ count: number of memory cells
            width: number of "bits" in a memory cell
            init_state: initial state of the memory, one of
			"randomized", "trainable", "fixed", 
                        "zero", or a predifined value. 
        """
        self._heads = []
        self._count = count
        self._width = width
        self._init_state = init_state

    def add_head(self, head, *args, **kwargs):
        # make a unique name for the head
        basename = kwargs.get("name", head.__name__)
        name = basename
        index = 1
        while name in [h.name for h in self._heads]:
            name = "%s_%i"%(basename, index)
            index += 1

        self._heads += [head(count=self._count, width=self._width, name=name, *args, **kwargs)]

    @property
    def read_heads(self):
        return [h for h in self._heads if h.output_size is not None]

    @property
    def head_states(self):
        all_states = []
        for h in self._heads:
            all_states += h.states
        return _merge_states(all_states)

    @property
    def state_size(self):
        readout_sizes = tuple(h.output_size for h in self.read_heads)
        mem_size = (self._count, self._width)
        head_states = tuple(s.shape for s in self.head_states)
        return readout_sizes, (mem_size, head_states)

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

        elif self._init_state == "zero":
            single_memory = tf.zeros((1, self._count, self._width), dtype=dtype)
            memory_state  = tf.tile(single_memory, tf.stack([batch_size, 1, 1]))

        else:
            memory_state  = tf.tile(self._init_state, tf.stack([batch_size, 1, 1]))

        readouts         = [tf.zeros((batch_size, h.output_size), dtype) for h in self.read_heads]
        private          = tuple(s.zero_state(batch_size, dtype) for s in self.head_states)
        
        return readouts, (memory_state, private)

    def _prepare_heads(self, controller_vec, memory_state, head_states):
        """ calculates commands of all heads.
        """
        # build state dictionary
        current_head_states = {}
        for (i, n) in enumerate(self.head_states):
            current_head_states[n.name] = head_states[i]

        commands = []
        for (i, head) in enumerate(self._heads):
            with tf.variable_scope("head_%i"%i):
                interface_vector = tf.layers.dense(controller_vec, 
                                    head.input_size, 
                                    use_bias = False)

                command, state = head.command(interface_vector,
                                              memory_state,
                                              current_head_states)
                commands += [command]
                for n in state:
                    current_head_states[n] = state[n]

        # build the state tuple from the dictionary
        new_head_states = tuple(current_head_states[h.name] for h in self.head_states)

        return tuple(commands), new_head_states

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
        l += [h.shape for h in self.head_states]
        return tuple(l)

    def pack_summary(self, memory_state):
        flat = list(map(_flatten_to_2d, memory_state[1]))
        return tf.concat([_flatten_to_2d(memory_state[0])]+flat, axis=1)

    def unpack_summary(self, summary):
        batch_size = tf.shape(summary)[0]
        time_size = tf.shape(summary)[1]
        ss = list(map(np.prod, self.summary_shape))
        split = tf.split(summary, ss, axis = 2)
        reshaped = [tf.reshape(data, [batch_size, time_size]+list(shape)) for (data, shape) in zip(split, self.summary_shape)]
        names = ["memory"] + [h.name for h in self.head_states]
        return {n:v for (n, v) in zip(names, reshaped)}


class HeadState:
    def __init__(self, shape, name, head):
        self._shape = shape
        self._head = head
        if not name.startswith("/"):
            name = head.name + "/" + name
        self._name = name

    @property
    def shape(self):
        return self._shape

    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size]+list(self.shape), dtype=dtype)

    @property
    def name(self):
        return self._name

    def get_value_from(self, dict):
        return dict[self.name]

class Head:
    def __init__(self, in_size, out_size, name):
        self._in_size  = in_size
        self._out_size = out_size
        self._name     = name
        self._states   = []


    def _add_state(self, name, shape, stateclass=HeadState):
        state = stateclass(name=name, shape=shape, head=self)
        self._states += [state]
        return state

    def command(self, control, memory, state):
        raise NotImplementedError()

    def execute(self, memory, commands):
        raise NotImplementedError()

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    @property
    def states(self):
        return self._states

    @property
    def name(self):
        return self._name

