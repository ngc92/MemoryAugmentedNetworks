import numpy as np
from .task import Task

def random_symbol(n, bits, vectors):
    symbol = np.zeros((n, vectors+1, bits+2))
    random = np.random.randint(2, size=(n,vectors,bits))
    symbol[:,1:,:-2] = random
    symbol[:,0,-2] = 1
    return symbol


class RecallTask(Task):
    def __init__(self, bits, vectors, symbols):
        super(RecallTask, self).__init__(bits+2, bits+2)
        self._bits = bits
        self._vectors = vectors
        self._symbols = symbols

    @property
    def default_params(self):
        return (self._symbols, self._vectors)

    def __call__(self, n, symbols=None, vectors=None):
        symbols = self._get_value("symbols", symbols)
        vectors = self._get_value("vectors", vectors)

        request = np.random.randint(symbols-1)
        symbol_list = [random_symbol(n,self._bits, vectors) for i in range(symbols)]
        symbol_list.append(symbol_list[request])

        input_seq  = np.zeros((n, (symbols+2)*(vectors+1), self._bits+2))
        output_seq = np.zeros((n, (symbols+2)*(vectors+1), self._bits+2))
        mask       = np.zeros((n, (symbols+2)*(vectors+1), self._bits+2))

        input_seq[:, :(symbols+1)*(vectors+1), :] = np.concatenate(np.array(symbol_list),axis =1)
        input_seq[:, symbols*(vectors+1), -2] = 0
        input_seq[:, symbols*(vectors+1), -1] = 1
        input_seq[:, (symbols+1)*(vectors+1), -1] = 1
        output_seq[:, (symbols+1)*(vectors+1)+1:, :-2] = symbol_list[request+1][:, 1:, :-2]
        mask[:, (symbols+1)*(vectors+1)+1:, :-2] = 1
        
        return input_seq, output_seq, mask