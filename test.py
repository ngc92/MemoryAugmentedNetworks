import tensorflow as tf

from dnc import DNC, LSTMCell
from dnc.memory import NTMMemory

from tasks.copy import CopyTask

INPUT_SIZE = 8

memory = NTMMemory(20, 80)
rheads = memory.read_heads(5)
wheads = memory.write_heads(1)

input = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))
lstm  = LSTMCell(25)

net = DNC(input, memory, rheads, wheads, INPUT_SIZE, controller = lstm)
targets = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
loss = tf.losses.mean_squared_error(net[0], targets)

opt = tf.train.AdamOptimizer(1e-3)
train = opt.minimize(loss)


task = CopyTask(2**INPUT_SIZE-1, 10)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1000):
        training_set = task(64)
        l, o = session.run([loss, train], feed_dict={input:training_set[0], targets: training_set[1]})
        print(l)

