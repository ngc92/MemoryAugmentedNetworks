import tensorflow as tf
LSTMCell = tf.nn.rnn_cell.LSTMCell

from .cell import DNCCell

def DNC(input, *args, controller = LSTMCell(50), dtype=tf.float32, log_memory=False, **kwargs):
    dnccell  = DNCCell(controller, log_memory=log_memory, *args, **kwargs)
    outputs, state = tf.nn.dynamic_rnn(dnccell, input, dtype=dtype)
    if log_memory:
        # dissect outputs
        out        = outputs[:, :, :dnccell.raw_output_size]
        summarized = outputs[:, :, dnccell.raw_output_size:]
        return out, state, dnccell.unpack_summary(summarized) 
    else:
        return outputs, state, None
