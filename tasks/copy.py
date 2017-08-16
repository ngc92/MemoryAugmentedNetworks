import numpy as np

from .task import Task, to_binary

class CopyTask(Task):
    def __init__(self, bits, sequence_length):
        super(CopyTask, self).__init__(bits+2, bits+2)
        self._bits = bits
        self._sequence_length = sequence_length

    @property
    def default_params(self):
        return (self._sequence_length,)


    def __call__(self, n, sequence_length = None):
        sequence_length = self._get_value("sequence_length", sequence_length)

        # random sequence
        seq = np.random.randint(2**self._bits, size=(n, sequence_length))
        binaries = np.array(list(map(lambda x: to_binary(x, self._bits), seq)))

        input_seq  = np.zeros((n, sequence_length*2+2, self._bits+2))
        output_seq = np.zeros((n, sequence_length*2+2, self._bits+2))
        mask       = np.zeros((n, sequence_length*2+2, self._bits+2))

        input_seq[:, 1:sequence_length+1, :-2] = binaries
        output_seq[:, sequence_length+2:, :-2] = binaries
        mask[:, sequence_length+2:, :] = 1
        input_seq[:,0,-2] = 1
        input_seq[:,sequence_length+1,-1] = 1

        return input_seq, output_seq, mask
