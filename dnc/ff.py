
import tensorflow as tf


def simple_feedforward(hidden=None, output_nl=tf.tanh, activation=tf.nn.relu):
    if hidden is None: hidden = [300, 300]

    def ff(input):
        out = input
        for i in hidden:
            out = tf.layers.dense(out, i, activation=activation)
        return output_nl(out)

    return ff


class FFWrapper:

    def __init__(self, feedforward):
        self._feedforward = feedforward

    @property
    def state_size(self):
        return 0

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, 0), dtype=dtype)

    def __call__(self, input, state):
        return self._feedforward(input), state

