import tensorflow as tf

from dnc import DNC, LSTMCell
from dnc.memory import NTMMemory

from tasks.copy import CopyTask

INPUT_SIZE = 8

memory = NTMMemory(128, 20)
rheads = memory.read_heads(1)
wheads = memory.write_heads(1)

input = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))
lstm  = LSTMCell(100)

net = DNC(input, memory, rheads, wheads, INPUT_SIZE, controller = lstm)
targets = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
mask    = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
loss = tf.losses.sigmoid_cross_entropy(logits=net[0], multi_class_labels=targets, weights=mask)
cost = tf.reduce_sum((1 - targets * (1 - tf.exp(-net[0]))) * tf.sigmoid(net[0]) * mask)

opt = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
train = opt.minimize(loss)


task = CopyTask(2**INPUT_SIZE-1, 10)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(200*1000):
        training_set = task(64)
        l, o, c = session.run([loss, train, cost], feed_dict={input:training_set[0], 
        													  targets: training_set[1], 
        													  mask: training_set[2]})
        print(i * 64 // 1000, l, c/64)

