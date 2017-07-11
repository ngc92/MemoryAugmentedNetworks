import numpy as np
from .task import Task, to_binary

class RepeatCopyTask:
    def __init__(self, bits, sequence_length, repeats):
        super(RepeatCopyTask, self).__init__(bits+2, bits+1)
        self._bits = bits
        self._sequence_length = sequence_length
        self._repeats = repeats

    def __call__(self, n, sequence_length=None, repeats=None):
        if sequence_length is None:
            sequence_length = self._sequence_length
        if repeats is None:
            repeats = self._repeats
        # random sequence
        seq = np.random.randint(2**self._bits, size=(n, sequence_length))
        binaries = np.array(list(map(lambda x: to_binary(x, self._bits), seq)))

        input_seq  = np.zeros((n, sequence_length*(repeats+1)+3, self._bits+2))
        output_seq = np.zeros((n, sequence_length*(repeats+1)+3, self._bits+1))

        input_seq[:, 1:sequence_length+1, :-2] = binaries        
        input_seq[:, 0,-2] = 1
        input_seq[:, sequence_length+1,-1] = repeats

        (a,b,c) = binaries.shape
        output_seq[:, sequence_length+2:, :-1] = np.repeat(binaries, repeats, axis=0).reshape(a,repeats*b,c)
        output_seq[:, -1, -1] = 1 # end marker

        return input_seq, output_seq
