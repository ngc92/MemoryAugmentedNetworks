import tensorflow as tf
from collections import namedtuple
RNNCell = tf.nn.rnn_cell.RNNCell

DNCStateTuple = namedtuple("DNCStateTuple", 
                          ( "readouts", 
                            "controller_state", 
                            "memory_state", 
                            "write_states", 
                            "read_states" 
                          ))


class DNCCell(RNNCell):
  """The basic cell for a DNC operation.
  """

  def __init__( self, controller, memory, read_heads, 
                write_heads, output_size, reuse=None ):

    super(DNCCell, self).__init__(_reuse=reuse)
    self._controller   = controller
    self._memory       = memory
    self._read_heads   = read_heads
    self._write_heads  = write_heads
    self._output_size  = output_size

  @property
  def state_size(self):
    readout_sizes     = []
    read_state_sizes  = []
    write_state_sizes = []

    for head in self._read_heads:
        readout_sizes += [head.output_size]
        read_state_sizes += [head.state_size]

    for head in self._write_heads:
        write_state_sizes += [head.state_size]

    size = ( tuple(readout_sizes), 
             self._controller.state_size, 
             self._memory.state_size, 
             tuple(write_state_sizes), 
             tuple(read_state_sizes) )

    return size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    readouts         = [tf.zeros((batch_size, h.output_size), dtype) for h in self._read_heads]
    controller_state = self._controller.zero_state(batch_size, dtype)
    memory_state     = self._memory.zero_state(batch_size, dtype)
    write_states     = [h.zero_state(batch_size, dtype) for h in self._write_heads]
    read_states      = [h.zero_state(batch_size, dtype) for h in self._read_heads]

    return DNCStateTuple( readouts, 
                          controller_state, 
                          memory_state, 
                          write_states, 
                          read_states )

  def call(self, inputs, state):
    # sample inputs are to be one-dimensional
    assert len(inputs.shape) == 2

    # get controller input and apply controller
    cinput = tf.concat([inputs] + state.readouts, axis=1)
    coutput, controller_state = self._controller(cinput, state.controller_state)

    # actions of write heads
    write_states = []
    for (i, head) in enumerate(self._write_heads):
        with tf.variable_scope("write_interface"):
            # initializer?
            cvec = tf.layers.dense(coutput, head.input_size, use_bias = False)
            memory_state, write_state = head(cvec, state.memory_state, state.write_states[i])
            write_states.append(write_state)

    # actions of read heads
    readouts    = []
    read_states = []
    for (i, head) in enumerate(self._read_heads):
        with tf.variable_scope("read_interface_%i"%i):
            # initializer?
            cvec = tf.layers.dense(coutput, head.input_size, use_bias = False)
            readout, read_state = head(cvec, state.memory_state, state.read_states[i])
            read_states.append(read_state)
            readouts.append(readout)


    # collect updated state
    new_state = DNCStateTuple( readouts,
                               controller_state, 
                               memory_state, 
                               write_states,
                               read_states ) 

    # combine readouts and output to total output
    controller_out = tf.layers.dense(coutput, self.output_size, use_bias = False)
    all_readouts   = tf.concat(readouts, axis=1)
    readout_out    = tf.layers.dense(all_readouts, self.output_size, use_bias = False)
    total_out      = controller_out + readout_out

    return total_out, new_state
