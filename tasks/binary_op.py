import numpy as np
from .task import Task

def random_symbol(n, bits, vectors):
    symbol = np.zeros((n, vectors+1, bits+2), dtype=np.int)
    random = np.random.randint(2, size=(n,vectors,bits))
    symbol[:,1:,:-2] = random
    symbol[:,0,-2] = 1
    return symbol


def xor_op(a, b):
    return a^b

def and_op(a, b):
    return a*b

def merge_op(x, y):
    (a,b,c) = x.shape
    output = np.zeros((a,b,c))
    output[:, :, :c//2] = x[:, :, :c//2]
    output[:, :, c//2:] = y[:, :, c//2:]
    return output

class BinaryOpTask(Task):
    def __init__(self, bits, vectors, binary_op):
        super(BinaryOpTask, self).__init__(bits+2, bits+2)
        self._bits = bits
        self._vectors = vectors
        self._binary_op = binary_op

    def __call__(self, n, vectors=None):
        if vectors is None:
            vectors = self._vectors
        symbol_list = [random_symbol(n,self._bits,vectors) for i in range(2)]
        target = self._binary_op(symbol_list[0][:, 1:, :-2], symbol_list[1][:, 1:, :-2])

        input_seq  = np.zeros((n, 3*(vectors+1), self._bits+2))
        output_seq = np.zeros((n, 3*(vectors+1), self._bits+2))
        mask       = np.zeros((n, 3*(vectors+1), self._bits+2))

        input_seq[:, :2*(vectors+1), :] = np.concatenate(np.array(symbol_list),axis =1)
        input_seq[:, 2*(vectors+1), -1] = 1
        output_seq[:, 2*(vectors+1)+1:, :-2] = target
        mask[:, 2*(vectors+1)+1:, :-2] = 1
        
        return input_seq, output_seq, mask

class XorTask(BinaryOpTask):
    def __init__(self, bits, vectors):
        super(XorTask, self).__init__(bits, vectors, xor_op)

class AndTask(BinaryOpTask):
    def __init__(self, bits, vectors):
        super(AndTask, self).__init__(bits, vectors, and_op)

class MergeTask(BinaryOpTask):
    def __init__(self, bits, vectors):
        super(MergeTask, self).__init__(bits, vectors, merge_op)
