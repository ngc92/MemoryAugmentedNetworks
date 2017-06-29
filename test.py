import tensorflow as tf

from dnc import DNC, LSTMCell
from dnc.memory import NTMMemory

from tasks.copy import CopyTask
import numpy as np
import tensorflow as tf

INPUT_SIZE = 8
BATCH_SIZE = 32

memory = NTMMemory(20, 128)
rheads = memory.read_heads(0)
wheads = memory.write_heads(0)

input = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))
lstm  = tf.nn.rnn_cell.MultiRNNCell([LSTMCell(256) for i in range(3)])

net = DNC(input, memory, rheads, wheads, INPUT_SIZE, controller = lstm)
targets = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
mask    = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
loss = tf.losses.sigmoid_cross_entropy(logits=net[0], multi_class_labels=targets, weights=mask)
cost = tf.reduce_sum((1 - targets * (1 - tf.exp(-net[0]))) * tf.sigmoid(net[0]) * mask)

opt = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
train = opt.minimize(loss)


task = CopyTask(2**INPUT_SIZE-1, 10)

pcount = 0
for v in tf.trainable_variables():
    pcount += np.product(list(map(lambda x: x.value, v.shape)))
print(pcount)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(200*1000):
        lg = np.random.randint(19) + 1
        training_set = task(BATCH_SIZE, lg)
        l, o, c = session.run([loss, train, cost], feed_dict={input:training_set[0], 
                                                              targets: training_set[1], 
                                                              mask: training_set[2]})
        print(i * BATCH_SIZE // 1000, l, c/BATCH_SIZE)

