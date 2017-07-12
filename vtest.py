
import tensorflow as tf
import numpy as np

import dnc
from dnc import DNC, LSTMCell
from dnc.memory import Memory, NTMReadHead, NTMWriteHead

from tasks import *
from utils import *

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--msize", type=int, default=128)
parser.add_argument("--mwidth", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--controller-size", type=int, default=100)
parser.add_argument("--minit", type=str, default="randomized")
parser.add_argument("--task", type=str, default="CopyTask(8, (1, 20))")
parser.add_argument("--controller", type=str, choices=["lstm", "multilstm", "ff"], default="lstm")
parser.add_argument("--no-dnc", action='store_true')
parser.add_argument("--test-scale", type=float, default=2)
parser.add_argument("--savedir", type=str, default="model")
parser.add_argument("--logdir",  type=str, default="logs")
args = parser.parse_args()

BATCH_SIZE = args.batch_size

task = eval(args.task)

memory = Memory(args.msize, args.mwidth, init_state=args.minit)
memory.add_head(NTMReadHead, shifts=[-1, 0, 1])
memory.add_head(NTMWriteHead, shifts=[-1, 0, 1])

input = tf.placeholder(tf.float32, shape=(None, None, task.input_size))
#
if args.controller == 'lstm':
    controller  = LSTMCell(args.controller_size)
elif args.controller == 'multilstm':
    controller  = tf.nn.rnn_cell.MultiRNNCell([LSTMCell(args.controller_size) for i in range(3)])
elif args.controller == 'ff':
    controller = dnc.ff.FFWrapper(dnc.ff.simple_feedforward(hidden=[args.controller_size]*2))

if not args.no_dnc:
    net = DNC(input, memory, output_size=task.output_size, controller = controller, log_memory=True)
    output  = net[0]
else:
    output, _ = tf.nn.dynamic_rnn(controller, input, dtype=tf.float32)
    output = tf.layers.dense(output, task.output_size, use_bias=False)

targets = tf.placeholder(dtype=tf.float32, shape=[None, None, task.output_size])
loss = tf.losses.sigmoid_cross_entropy(logits=output, multi_class_labels=targets)
cost = tf.reduce_sum((1 - targets * (1 - tf.exp(-output))) * tf.sigmoid(output)) / BATCH_SIZE

opt = tf.train.RMSPropOptimizer(1e-4, momentum=0.9)
train = minimize_and_clip(opt, loss)

img_summary =  [tf.summary.image("IO/input", concate_to_image(input), max_outputs=1)]
img_summary += [tf.summary.image("IO/targets", concate_to_image(targets), max_outputs=1)]
img_summary += [tf.summary.image("IO/output", tf.sigmoid(concate_to_image(output)), max_outputs=1)]
if not args.no_dnc:
    img_summary += [tf.summary.image(key, concate_to_image(net[2][key]), max_outputs=1) for key in net[2]]
img_summary = tf.summary.merge(img_summary)
scalar_summary = [tf.summary.scalar("train/cost", cost), tf.summary.scalar("train/loss", loss)]
scalar_summary += [tf.summary.scalar(name, value) for (name, value) in weight_norms()]
scalar_summary = tf.summary.merge(scalar_summary)

scalar_test_summary = tf.summary.merge([tf.summary.scalar("test/cost", cost), tf.summary.scalar("test/loss", loss)])

pcount = 0
for v in tf.trainable_variables():
    pcount += np.product(list(map(lambda x: x.value, v.shape)))
print("Number of parameters: %i"%pcount)

w = tf.summary.FileWriter(args.logdir)
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(200*1000):
        if i % 100 != 0:    # training
            training_set = task(BATCH_SIZE)
            l, o, c, s2 = session.run([loss, train, cost, scalar_summary],
                                            feed_dict={ input:training_set[0],
                                                        targets: training_set[1]})

            w.add_summary(s2, global_step=i*BATCH_SIZE)
        else: # testing
            def make_big(data):
                if isinstance(data, tuple):
                    return int(data[1]*args.test_scale)
                else:
                    return int(data * args.test_scale)
            
            defp = map(make_big, task.default_params)
            training_set = task(BATCH_SIZE, *defp)
            l, c, s1, s2 = session.run([loss, cost, img_summary, scalar_test_summary],
                                            feed_dict={ input:training_set[0],
                                                        targets: training_set[1]})

            w.add_summary(s1, global_step=i*BATCH_SIZE)
            w.add_summary(s2, global_step=i*BATCH_SIZE)
            print(i * BATCH_SIZE / 1000, l, c)
            saver.save(session, os.path.join(args.savedir, 'model'), global_step=i)
