
import tensorflow as tf
import numpy as np

from dnc import DNC, LSTMCell
from dnc.memory import Memory, NTMReadHead, NTMWriteHead

from tasks import CopyTask, RepeatCopyTask
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--msize", type=int, default=128)
parser.add_argument("--mwidth", type=int, default=16)
parser.add_argument("--bsize", type=int, default=32)
parser.add_argument("--isize", type=int, default=8)
parser.add_argument("--rsize", type=int, default=100)
parser.add_argument("--minit", type=str, default="randomized")
args = parser.parse_args()

INPUT_SIZE = args.isize
BATCH_SIZE = args.bsize

memory = Memory(args.msize, args.mwidth, init_state=args.minit)
memory.add_head(NTMReadHead, shifts=[-1, 0, 1])
memory.add_head(NTMWriteHead, shifts=[-1, 0, 1])

input = tf.placeholder(tf.float32, shape=(None, None, INPUT_SIZE))
#lstm  = tf.nn.rnn_cell.MultiRNNCell([LSTMCell(256) for i in range(3)])
lstm  = LSTMCell(args.rsize)

net = DNC(input, memory, INPUT_SIZE, controller = lstm, log_memory=True)
targets = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
mask    = tf.placeholder(dtype=tf.float32, shape=[None, None, INPUT_SIZE])
output  = net[0]
loss = tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=targets)
cost = tf.reduce_sum((1 - targets * (1 - tf.exp(-output))) * tf.sigmoid(output)) / BATCH_SIZE

opt = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
train = minimize_and_clip(opt, loss)

print(net[2])
names = ["memory", "read", "write"]
img_summary = [tf.summary.image(name, concate_to_image(data), max_outputs=1) for (name, data) in zip(names, net[2])]
img_summary = tf.summary.merge(img_summary)
scalar_summary = [tf.summary.scalar("cost", cost), tf.summary.scalar("loss", loss)]
scalar_summary += [tf.summary.scalar(name, value) for (name, value) in weight_norms()]
scalar_summary = tf.summary.merge(scalar_summary)

task = CopyTask(2**INPUT_SIZE-1, 10)

pcount = 0
for v in tf.trainable_variables():
    pcount += np.product(list(map(lambda x: x.value, v.shape)))
print(pcount)

w = tf.summary.FileWriter("logs")

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(200*1000):

        lg = np.random.randint(19) + 1
        if i % 100 == 0:
            lg = 20
        training_set = task(BATCH_SIZE, lg)
        l, o, c, s1, s2 = session.run([loss, train, cost, img_summary, scalar_summary],
                                        feed_dict={ input:training_set[0],
                                                    targets: training_set[1], 
                                                    mask: training_set[2]})

        w.add_summary(s2, global_step=i*BATCH_SIZE)
        if i % 100 == 0:
            w.add_summary(s1, global_step=i*BATCH_SIZE)
        print(i * BATCH_SIZE / 1000, l, c)
