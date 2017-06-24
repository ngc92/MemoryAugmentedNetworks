import tensorflow as tf
LSTMCell = tf.nn.rnn_cell.LSTMCell

from .cell import DNCCell

def DNC(input, *args, controller = LSTMCell(50), dtype=tf.float32, **kwargs):
    dnccell  = DNCCell(controller, *args, **kwargs)
    return tf.nn.dynamic_rnn(dnccell, input, dtype=dtype)
