import tensorflow as tf
from collections import namedtuple
RNNCell = tf.nn.rnn_cell.RNNCell

from memory import SimpleMemory


DNCStateTuple = namedtuple("DNCStateTuple", ("controller_state", "read", "memory"))

class DNCCell(RNNCell):
  """The basic cell for a DNC operation.
  """

  def __init__(self, output_size, controller_cell, memory, read_heads, write_heads, reuse=None):
    super(DNCCell, self).__init__(_reuse=reuse)
    self._output_size     = output_size
    self._controller_cell = controller_cell
    self._read_heads      = read_heads
    self._write_heads     = write_heads
    self._memory          = memory

  @property
  def state_size(self):
    # need to remember all previously read data
    read_sizes = []
    for head in self._read_heads:
        read_sizes += [head.output_size]
    size = (self._controller_cell.state_size, tuple(read_sizes), self._memory.state_size)
    return size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    read_zeros = [tf.zeros((batch_size, h.output_size), dtype=dtype) for h in self._read_heads]
    return DNCStateTuple(self._controller_cell.zero_state(batch_size, dtype), 
        tuple(read_zeros), 
        self._memory.zero_state(batch_size, dtype))

  def call(self, inputs, state):
    assert len(inputs.shape) == 2
    # feed the controller
    cinput = tf.concat([inputs]+list(state.read), axis=1)

    memory = state.memory

    # apply the controller
    controller_out, controller_state = self._controller_cell(cinput, state.controller_state)

    readouts = []
    # calculate interface vectors for each read_head
    for (i, head) in enumerate(self._read_heads):
        with tf.variable_scope("read_interface_%i"%i):
            # initializer?
            cvec = tf.layers.dense(controller_out, head.input_size, use_bias = False)
            readouts += [head(cvec, memory)]

    # write heads
    for (i, head) in enumerate(self._write_heads):
        with tf.variable_scope("write_interface_%i"%i):
            # initializer?
            cvec = tf.layers.dense(controller_out, head.input_size, use_bias = False)
            memory = head(cvec, memory)

    new_state = DNCStateTuple(controller_state = controller_state, read = tuple(readouts), memory = memory)

    # combine readouts and output
    controller_out = tf.layers.dense(controller_out, self.output_size, use_bias = False)
    all_readouts   = tf.concat(readouts, axis=1)
    readout_out    = tf.layers.dense(all_readouts, self.output_size, use_bias = False)
    total_out      = controller_out + readout_out

    return total_out, new_state


input = tf.placeholder(tf.float32, shape=(10, 5, 5))
lstm = tf.nn.rnn_cell.LSTMCell(25)
memory = SimpleMemory(12, 42)
dnc = DNCCell(64, lstm, memory, [memory.read_head(), memory.read_head()], [memory.write_head()])

tf.nn.dynamic_rnn(dnc, input, dtype=tf.float32)

writer = tf.summary.FileWriter("logs/test", graph=tf.get_default_graph())
