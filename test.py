import tensorflow as tf
import numpy as np

from dnc import DNC, LSTMCell
from dnc.memory import Memory, NTMReadHead, NTMWriteHead

from tasks import CopyTask
from utils import *

INPUT_SIZE = 8
BATCH_SIZE = 32

memory = Memory(25, 20)
memory.add_head(NTMReadHead, shifts=[-1, 0, 1])
memory.add_head(NTMWriteHead, shifts=[-1, 0, 1])

input = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))
#lstm  = tf.nn.rnn_cell.MultiRNNCell([LSTMCell(256) for i in range(3)])
lstm  = LSTMCell(100)

net = DNC(input, memory, INPUT_SIZE, controller = lstm, log_memory=True)
targets = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
mask    = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
output  = net[0]
loss = tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=targets)
cost = tf.reduce_sum((1 - targets * (1 - tf.exp(-output))) * tf.sigmoid(output))

opt = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
train = minimize_and_clip(opt, loss)

print(net[2])
names = ["memory", "read", "write"]
img_summary = [tf.summary.image(name, concate_to_image(data), max_outputs=1) for (name, data) in zip(names, net[2])]
img_summary = tf.summary.merge(img_summary)


task = CopyTask(2**INPUT_SIZE-1, 10)

pcount = 0
for v in tf.trainable_variables():
    pcount += np.product(list(map(lambda x: x.value, v.shape)))
print(pcount)

w = tf.summary.FileWriter("logs")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(200*1000):

        lg = 10#np.random.randint(19) + 1
        if i % 100 == 0:
            lg = 20
        training_set = task(BATCH_SIZE, lg)
        l, o, c, s = session.run([loss, train, cost, img_summary], feed_dict={input:training_set[0], 
                                                              targets: training_set[1], 
                                                              mask: training_set[2]})

        if i % 100 == 0:
            w.add_summary(s, global_step=i)
        print(i * BATCH_SIZE // 1000, l, c/BATCH_SIZE)
