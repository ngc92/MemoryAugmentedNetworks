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

    @property
    def default_params(self):
        return tuple()

    def _get_value(self, name, supplied):
        if supplied is None:
            value = getattr(self, "_"+name)
        else:
            value = supplied
        if isinstance(value, tuple):
            return np.random.randint(value[1] - value[0]) + value[0]
        else:
            return value


# utility functions for tasks
def to_binary(x, l):
    vec = np.zeros((len(x), l))
    binary = list(map(lambda x: list(map(float, ("{0:0%ib}"%(l)).format(x))), x))
    vec[:, :] = binary
    return vec