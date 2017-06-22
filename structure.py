import tensorflow as tf
from collections import namedtuple
RNNCell = tf.nn.rnn_cell.RNNCell


DNCStateTuple = namedtuple("DNCStateTuple", ("controller_state", "read", "memory"))

class ReadHead:
    def __init__(self):
        pass

    @property
    def input_size(self):
        pass

    @property
    def output_size(self):
        pass

    def __call__(self, control):
        pass
        # returns read data

class DummyRead(ReadHead):
    def __init__(self, in_size, out_size):
        self._in_size = in_size
        self._out_size = out_size

    @property
    def input_size(self):
        return self._in_size

    @property
    def output_size(self):
        return self._out_size

    def __call__(self, control, memory):
        with tf.name_scope("DummyRead"):
            return tf.zeros((tf.shape(control)[0], self.output_size))

class WriteHead:
    def __init__(self):
        pass

    @property
    def input_size(self):
        return 10

    def __call__(self, control, memory):
        return memory

class DNCCell(RNNCell):
  """The basic cell for a DNC operation.
  """

  def __init__(self, output_size, controller_cell, read_heads, write_heads, reuse=None):
    super(DNCCell, self).__init__(_reuse=reuse)
    self._output_size     = output_size
    self._controller_cell = controller_cell
    self._read_heads      = read_heads
    self._write_heads     = write_heads

  @property
  def state_size(self):
    # need to remember all previously read data
    read_sizes = []
    for head in self._read_heads:
        read_sizes += [head.output_size]
    return (self._controller_cell.state_size, tuple(read_sizes))

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    return DNCStateTuple(*super(DNCCell, self).zero_state(batch_size, dtype))

  def call(self, inputs, state):
    assert len(input.shape) == 2
    # feed the controller
    cinput = tf.concat([inputs]+list(state.read), axis=1)

    # apply the controller
    controller_out, controller_state = self._controller_cell(cinput, state.controller_state)

    readouts = []
    # calculate interface vectors for each read_head
    for (i, head) in enumerate(self._read_heads):
        with tf.variable_scope("read_interface_%i"%i):
            # initializer?
            cvec = tf.layers.dense(controller_out, head.input_size, use_bias = False)
            readouts += [head(cvec)]

    # write heads
    for (i, head) in enumerate(self._write_heads):
        with tf.variable_scope("write_interface_%i"%i):
            # initializer?
            cvec = tf.layers.dense(controller_out, head.input_size, use_bias = False)
            head(cvec)

    new_state = DNCStateTuple(controller_state=controller_state, read=readouts)

    return controller_out, new_state



input = tf.placeholder(tf.float32, shape=(10, 5))
lstm = tf.nn.rnn_cell.LSTMCell(25)
zero_read = DummyRead(5, 42)
dnc = DNCCell(64, lstm, [zero_read, zero_read], [WriteHead()])

tf.nn.static_rnn(dnc, [input]*5, dtype=tf.float32)

writer = tf.summary.FileWriter("logs/test", graph=tf.get_default_graph())

