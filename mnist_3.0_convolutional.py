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

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets(train_dir="data", one_hot=True, reshape=False, validation_size=0)

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [5, 5, 1, 4]        B1 [4]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 2        W2 [5, 5, 4, 8]        B2 [8]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 2       W3 [4, 4, 8, 12]       B3 [12]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# variable learning rate
lr = tf.placeholder(dtype=tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(
    initial_value=tf.truncated_normal(shape=[5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(initial_value=tf.ones(shape=[K]) / 10)
W2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, K, L], stddev=0.1))
B2 = tf.Variable(initial_value=tf.ones(shape=[L]) / 10)
W3 = tf.Variable(initial_value=tf.truncated_normal(shape=[4, 4, L, M], stddev=0.1))
B3 = tf.Variable(initial_value=tf.ones(shape=[M]) / 10)

W4 = tf.Variable(initial_value=tf.truncated_normal(shape=[7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(initial_value=tf.ones(shape=[N]) / 10)
W5 = tf.Variable(initial_value=tf.truncated_normal(shape=[N, 10], stddev=0.1))
B5 = tf.Variable(initial_value=tf.ones(shape=[10]) / 10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(features=tf.nn.conv2d(input=X, filter=W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(features=tf.nn.conv2d(input=Y1, filter=W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(features=tf.nn.conv2d(input=Y2, filter=W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(tensor=Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(features=tf.matmul(a=YY, b=W4) + B4)
Ylogits = tf.matmul(a=Y4, b=W5) + B5
Y = tf.nn.softmax(logits=Ylogits)

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
            tf.reshape(tensor=W4, shape=[-1]),
            tf.reshape(tensor=W5, shape=[-1])], axis=0)
allbiases = tf.concat(
    values=[tf.reshape(tensor=B1, shape=[-1]), tf.reshape(tensor=B2, shape=[-1]), tf.reshape(tensor=B3, shape=[-1]),
            tf.reshape(tensor=B4, shape=[-1]),
            tf.reshape(tensor=B5, shape=[-1])], axis=0)
I = tensorflowvisu.tf_format_mnist_images(X=X, Y=Y, Y_=Y_)
It = tensorflowvisu.tf_format_mnist_images(X=X, Y=Y, Y_=Y_, n=1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(fetches=init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(batch_size=100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run(fetches=[accuracy, cross_entropy, I, allweights, allbiases],
                                  feed_dict={X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(x=i, accuracy=a, loss=c)
        datavis.update_image1(im=im)
        datavis.append_data_histograms(x=i, datavect1=w, datavect2=b)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run(fetches=[accuracy, cross_entropy, It],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            c))
        datavis.append_test_curves_data(x=i, accuracy=a, loss=c)
        datavis.update_image2(im=im)

    # the backpropagation training step
    sess.run(fetches=train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate})


datavis.animate(compute_step=training_step, iterations=10001, train_data_update_freq=10, test_data_update_freq=100)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# layers 4 8 12 200, patches 5x5str1 5x5str2 4x4str2 best 0.989 after 10000 iterations
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 best 0.9892 after 10000 iterations
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 best 0.9908 after 10000 iterations but going downhill from 5000 on
# layers 6 12 24 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75 best 0.9922 after 10000 iterations (but above 0.99 after 1400 iterations only)
# layers 4 8 12 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9914 at 13700 iterations
# layers 9 16 25 200, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9918 at 10500 (but 0.99 at 1500 iterations already, 0.9915 at 5800)
# layers 9 16 25 300, patches 5x5str1 4x4str2 4x4str2 dropout=0.75, best 0.9916 at 5500 iterations (but 0.9903 at 1200 iterations already)
# attempts with 2 fully-connected layers: no better 300 and 100 neurons, dropout 0.75 and 0.5, 6x6 5x5 4x4 patches no better
# *layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 dropout=0.75 best 0.9928 after 12800 iterations (but consistently above 0.99 after 1300 iterations only, 0.9916 at 2300 iterations, 0.9921 at 5600, 0.9925 at 20000)
# layers 6 12 24 200, patches 6x6str1 5x5str2 4x4str2 no dropout best 0.9906 after 3100 iterations (avove 0.99 from iteration 1400)
