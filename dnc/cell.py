import tensorflow as tf
from collections import namedtuple
RNNCell = tf.nn.rnn_cell.RNNCell

DNCStateTuple = namedtuple("DNCStateTuple", 
                          ( "readouts", 
                            "controller_state", 
                            "memory_state"
                          ))


class DNCCell(RNNCell):
  """The basic cell for a DNC operation.
  """

  def __init__( self, controller, memory, output_size, reuse=None,
                output_nl = tf.identity ):

    super(DNCCell, self).__init__(_reuse=reuse)
    self._controller   = controller
    self._memory       = memory
    self._output_size  = output_size
    self._output_nl    = output_nl

  @property
  def state_size(self):
    readout_sizes, mem_state_size = self._memory.state_size
    size = ( readout_sizes, 
             self._controller.state_size, 
             mem_state_size)

    return size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    controller_state = self._controller.zero_state(batch_size, dtype)
    readouts, memory_state = self._memory.zero_state(batch_size, dtype)

    return DNCStateTuple( readouts, 
                          controller_state, 
                          memory_state )
  
  def call(self, inputs, state):
    # sample inputs are to be one-dimensional
    assert len(inputs.shape) == 2

    # get controller input and apply controller
    cinput = tf.concat([inputs] + state.readouts, axis=1)
    coutput, controller_state = self._controller(cinput, state.controller_state)

    # actions of write heads
    readouts, memory_state = self._memory(coutput, state.memory_state)

    # collect updated state
    new_state = DNCStateTuple( readouts,
                               controller_state, 
                               memory_state) 

    # combine readouts and output to total output
    controller_out = tf.layers.dense(coutput, self.output_size, use_bias = False)
    if len(readouts) != 0:
      all_readouts   = tf.concat(readouts, axis=1)
      readout_out    = tf.layers.dense(all_readouts, self.output_size, use_bias = False)
      total_out      = self._output_nl(controller_out + readout_out)
    else:
      print("Warning: Network does not read from memory!")
      total_out      = self._output_nl(controller_out)

    return total_out, new_state
