import tensorflow as tf

from dnc import DNC, LSTMCell
from dnc.memory import NTMMemory

memory = NTMMemory(20, 80)
rheads = memory.read_heads(5)
wheads = memory.write_heads(1)

input = tf.placeholder(tf.float32, shape=(10, None, 5))
lstm  = LSTMCell(25)

net = DNC(input, memory, rheads, wheads, 3, controller = lstm)
