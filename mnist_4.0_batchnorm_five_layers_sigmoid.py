# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid+BN)   W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid+BN)   W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid+BN)   W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid+BN)   W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax+BN)   W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets(train_dir="data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# variable learning rate
lr = tf.placeholder(dtype=tf.float32)
# train/test selector for batch normalisation
tst = tf.placeholder(dtype=tf.bool)
# training iteration
iter = tf.placeholder(dtype=tf.int32)

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
L = 200
M = 100
N = 60
P = 30
Q = 10

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(initial_value=tf.truncated_normal(shape=[784, L], stddev=0.1))  # 784 = 28 * 28
S1 = tf.Variable(initial_value=tf.ones(shape=[L]))
O1 = tf.Variable(initial_value=tf.zeros(shape=[L]))
W2 = tf.Variable(initial_value=tf.truncated_normal(shape=[L, M], stddev=0.1))
S2 = tf.Variable(initial_value=tf.ones(shape=[M]))
O2 = tf.Variable(initial_value=tf.zeros(shape=[M]))
W3 = tf.Variable(initial_value=tf.truncated_normal(shape=[M, N], stddev=0.1))
S3 = tf.Variable(initial_value=tf.ones(shape=[N]))
O3 = tf.Variable(initial_value=tf.zeros(shape=[N]))
W4 = tf.Variable(initial_value=tf.truncated_normal(shape=[N, P], stddev=0.1))
S4 = tf.Variable(initial_value=tf.ones(shape=[P]))
O4 = tf.Variable(initial_value=tf.zeros(shape=[P]))
W5 = tf.Variable(initial_value=tf.truncated_normal(shape=[P, Q], stddev=0.1))
B5 = tf.Variable(initial_value=tf.zeros(shape=[Q]))


## Batch normalisation conclusions with sigmoid activation function:
# BN is applied between logits and the activation function
# On Sigmoids it is very clear that without BN, the sigmoids saturate, with BN, they output
# a clean gaussian distribution of values, especially with high initial learning rates.

# sigmoid, no batch-norm, lr(0.003, 0.0001, 2000) => 97.5%
# sigmoid, batch-norm lr(0.03, 0.0001, 1000) => 98%
# sigmoid, batch-norm, no offsets => 97.3%
# sigmoid, batch-norm, no scales => 98.1% but cannot hold fast learning rate at start
# sigmoid, batch-norm, no scales, no offsets => 96%

# Both scales and offsets are useful with sigmoids.
# With RELUs, the scale variables can be omitted.
# Biases are not useful with batch norm, offsets are to be used instead

# Steady 98.5% accuracy using these parameters:
# moving average decay: 0.998 (equivalent to averaging over two epochs)
# learning rate decay from 0.03 to 0.0001 speed 1000 => max 98.59 at 6500 iterations, 98.54 at 10K it,  98% at 1300it, 98.5% at 3200it

def batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.998,
                                                       num_updates=iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(x=Ylogits, axes=[0])
    update_moving_averages = exp_moving_avg.apply(var_list=[mean, variance])
    m = tf.cond(pred=is_test, fn1=lambda: exp_moving_avg.average(mean), fn2=lambda: mean)
    v = tf.cond(pred=is_test, fn1=lambda: exp_moving_avg.average(variance), fn2=lambda: variance)
    Ybn = tf.nn.batch_normalization(x=Ylogits, mean=m, variance=v, offset=Offset, scale=Scale,
                                    variance_epsilon=bnepsilon)
    return Ybn, update_moving_averages


def no_batchnorm(Ylogits, Offset, Scale, is_test, iteration):
    return Ylogits, tf.no_op()


# The model
XX = tf.reshape(tensor=X, shape=[-1, 784])

Y1l = tf.matmul(XX, W1)
Y1bn, update_ema1 = batchnorm(Ylogits=Y1l, Offset=O1, Scale=S1, is_test=tst, iteration=iter)
Y1 = tf.nn.sigmoid(x=Y1bn)

Y2l = tf.matmul(a=Y1, b=W2)
Y2bn, update_ema2 = batchnorm(Ylogits=Y2l, Offset=O2, Scale=S2, is_test=tst, iteration=iter)
Y2 = tf.nn.sigmoid(x=Y2bn)

Y3l = tf.matmul(a=Y2, b=W3)
Y3bn, update_ema3 = batchnorm(Ylogits=Y3l, Offset=O3, Scale=S3, is_test=tst, iteration=iter)
Y3 = tf.nn.sigmoid(x=Y3bn)

Y4l = tf.matmul(a=Y3, b=W4)
Y4bn, update_ema4 = batchnorm(Ylogits=Y4l, Offset=O4, Scale=S4, is_test=tst, iteration=iter)
Y4 = tf.nn.sigmoid(x=Y4bn)

Ylogits = tf.matmul(a=Y4, b=W5) + B5
Y = tf.nn.softmax(logits=Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(input_tensor=cross_entropy) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(x=tf.argmax(input=Y, axis=1), y=tf.argmax(input=Y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32))

# matplotlib visualisation
allweights = tf.concat(
    values=[tf.reshape(tensor=W1, shape=[-1]), tf.reshape(tensor=W2, shape=[-1]), tf.reshape(tensor=W3, shape=[-1]),
            tf.reshape(tensor=W4, shape=[-1]), tf.reshape(tensor=W5, shape=[-1])], axis=0)
allbiases = tf.concat(
    values=[tf.reshape(tensor=O1, shape=[-1]), tf.reshape(tensor=O2, shape=[-1]), tf.reshape(tensor=O3, shape=[-1]),
            tf.reshape(tensor=O4, shape=[-1]), tf.reshape(tensor=B5, shape=[-1])], axis=0)
# to use for sigmoid
allactivations = tf.concat(
    values=[tf.reshape(tensor=Y1, shape=[-1]), tf.reshape(tensor=Y2, shape=[-1]), tf.reshape(tensor=Y3, shape=[-1]),
            tf.reshape(tensor=Y4, shape=[-1])], axis=0)
# to use for RELU
# allactivations = tf.concat([tf.reduce_max(Y1, [0]), tf.reduce_max(Y2, [0]), tf.reduce_max(Y3, [0]), tf.reduce_max(Y4, [0])], 0)
alllogits = tf.concat(
    values=[tf.reshape(tensor=Y1l, shape=[-1]), tf.reshape(tensor=Y2l, shape=[-1]), tf.reshape(tensor=Y3l, shape=[-1]),
            tf.reshape(tensor=Y4l, shape=[-1])], axis=0)
I = tensorflowvisu.tf_format_mnist_images(X=X, Y=Y, Y_=Y_)
It = tensorflowvisu.tf_format_mnist_images(X=X, Y=Y, Y_=Y_, n=1000, lines=25)
datavis = tensorflowvisu.MnistDataVis(title4="Logits", title5="activations", histogram4colornum=2, histogram5colornum=2)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(batch_size=100)

    # learning rate decay (without batch norm)
    # max_learning_rate = 0.003
    # min_learning_rate = 0.0001
    # decay_speed = 2000
    # learning rate decay (with batch norm)
    max_learning_rate = 0.03
    min_learning_rate = 0.0001
    decay_speed = 1000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, al, ac = sess.run(fetches=[accuracy, cross_entropy, I, alllogits, allactivations],
                                    feed_dict={X: batch_X, Y_: batch_Y, tst: False})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(x=i, accuracy=a, loss=c)
        datavis.update_image1(im=im)
        datavis.append_data_histograms(x=i, datavect1=al, datavect2=ac)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run(fetches=[accuracy, cross_entropy, It],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True})
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            c))
        datavis.append_test_curves_data(x=i, accuracy=a, loss=c)
        datavis.update_image2(im=im)

    # the backpropagation training step
    sess.run(fetches=train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False})
    sess.run(fetches=update_ema, feed_dict={X: batch_X, Y_: batch_Y, tst: False, iter: i})


datavis.animate(compute_step=training_step, iterations=10000 + 1, train_data_update_freq=20, test_data_update_freq=100,
                more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# Some results to expect:
# (In all runs, if sigmoids are used, all biases are initialised at 0, if RELUs are used,
# all biases are initialised at 0.1 apart from the last one which is initialised at 0.)

## decaying learning rate from 0.003 to 0.0001 decay_speed 2000, 10K iterations
# final test accuracy = 0.9813 (sigmoid - training cross-entropy not stabilised)
# final test accuracy = 0.9842 (relu - training set fully learned, test accuracy stable)
