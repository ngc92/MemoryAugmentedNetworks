
import tensorflow as tf

class SimpleMemory:
    def __init__(self, width, count):
        self._width = width
        self._count = count

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, self._count, self._width), dtype=dtype)

    @property
    def state_size(self):
        return (self._count, self._width)

    def read_head(self):
        return SimpleReadHead(self._count, self._width)

    def write_head(self):
        return SimpleWriteHead(self._count + self._width)


class SimpleReadHead:
    def __init__(self, in_size, out_size):
        self._in_size = in_size
        self._out_size = out_size

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    def state_size(self):
        return 0

    def zero_state(self, batch_size, dtype):
        return tf.zeros(0, dtype=dtype)

    def __call__(self, control, memory, state):
        # make control a probability vector
        probs = tf.nn.softmax(control)[:,:,None]
        read  = tf.reduce_sum(probs * memory, 1)
        return read, state


class SimpleWriteHead:
    def __init__(self, input_size):
        self._in_size = input_size

    @property
    def input_size(self):
        return self._in_size

    def state_size(self):
        return 0

    def zero_state(self, batch_size, dtype):
        return tf.zeros(0, dtype=dtype)

    def __call__(self, control, memory, state):
        mem_count = memory.shape[1]#tf.shape(memory)[1]
        p_part = control[:, 0:mem_count]
        d_part = control[:, mem_count:]
        probs = tf.expand_dims(tf.nn.softmax(p_part), -1)
        new_data = tf.expand_dims(d_part, 1)
        return memory + probs * new_data, state

