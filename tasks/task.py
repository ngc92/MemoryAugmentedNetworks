import numpy as np

class Task:
    def __init__(self, in_size, out_size):
        self._input_size  = in_size
        self._output_size = out_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size


# utility functions for tasks
def to_binary(x, l):
    vec = np.zeros((len(x), l))
    binary = list(map(lambda x: list(map(float, ("{0:0%ib}"%(l)).format(x))), x))
    vec[:, :] = binary
    return vec