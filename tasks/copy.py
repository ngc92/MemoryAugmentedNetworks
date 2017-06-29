import numpy as np

def to_binary(x, l):
    vec = np.zeros((len(x), l))
    binary = list(map(lambda x: list(map(float, ("{0:0%ib}"%(l)).format(x))), x))
    vec[:, :] = binary
    return vec

class CopyTask:
    def __init__(self, alphabet, sequence_length):
        self._alphabet = alphabet
        self._sequence_length = sequence_length

    def __call__(self, n):
        # random sequence
        vec_size = int(np.ceil(np.log(self._alphabet + 1) / np.log(2)))
        seq = np.random.randint(self._alphabet, size=(n, self._sequence_length)) + 1
        binaries = np.array(list(map(lambda x: to_binary(x, vec_size), seq)))

        input_seq  = np.zeros((n, self._sequence_length*2+1, vec_size))
        output_seq = np.zeros((n, self._sequence_length*2+1, vec_size))
        mask_seq   = np.zeros((n, self._sequence_length*2+1, vec_size))

        input_seq[:, 0:binaries.shape[1], :]   = binaries
        output_seq[:, binaries.shape[1]:-1, :] = binaries 
        mask_seq[:, binaries.shape[1]:-1, :]   = 1 
        
        return input_seq, output_seq, mask_seq
