import numpy as np

def to_binary(x, l):
    vec = np.zeros((len(x), l))
    binary = list(map(lambda x: list(map(float, ("{0:0%ib}"%(l)).format(x))), x))
    vec[:, :] = binary
    return vec

class RepeatCopyTask:
    def __init__(self, alphabet, sequence_length, repeats):
        self._alphabet = alphabet
        self._sequence_length = sequence_length
        self._repeats = repeats

    def __call__(self, n, sequence_length = None, repeats = None):
        if sequence_length is None:
            sequence_length = self._sequence_length
        if repeats is None:
            repeats = self._repeats
        # random sequence
        vec_size = int(np.ceil(np.log(self._alphabet + 1) / np.log(2)))
        seq = np.random.randint(self._alphabet, size=(n, sequence_length)) + 1
        binaries = np.array(list(map(lambda x: to_binary(x, vec_size), seq)))

        input_seq  = np.zeros((n, sequence_length*(repeats+1)+1, vec_size))
        output_seq = np.zeros((n, sequence_length*(repeats+1)+1, vec_size))

        length = binaries.shape[1]
        input_seq[:, 0:length, :]   = binaries
        (a,b,c) = binaries.shape
        output_seq[:, length:-1, :] = np.repeat(binaries, repeats, axis=0).reshape(a,repeats*b,c)
        
        return input_seq, output_seq