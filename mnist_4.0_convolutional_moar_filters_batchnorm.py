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
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from math import exp
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

##### helper functions #####
def init_conv_layer(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.ones([shape[3]])/10)

    return W, b


def init_fc_layer(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.ones([shape[1]]) / 10)

    return W, b


def def_fc_layer(input, weights, biases, keep_prob=1.0):
    logits = tf.matmul(input, weights) + biases
    A = tf.nn.dropout(tf.nn.relu(logits), keep_prob=keep_prob)

    return logits, A


def def_conv_layer(input, biases, filter, strides, padding='SAME'):
    conv = tf.nn.conv2d(input, filter=filter, strides=strides, padding=padding) + biases
    A = tf.nn.relu(conv)

    return A


def def_fc_layer_bn(input, weights, offset, is_training, iter, keep_prob):
    # no 'scale' (alpha) param if relu, no biases too, because we have scale
    ema = tf.train.ExponentialMovingAverage(wavg_decay, iter)

    logits = tf.matmul(input, weights)
    batch_mean, batch_var = tf.nn.moments(logits, [0])
    update_ema = ema.apply([batch_mean, batch_var])

    m = tf.cond(is_training, lambda: batch_mean, lambda: ema.average(batch_mean))
    v = tf.cond(is_training, lambda: batch_var, lambda: ema.average(batch_var))

    bn = tf.nn.batch_normalization(logits, m, v, offset, None, variance_epsilon=1e-8)
    A = tf.nn.relu(bn)
    Ad = tf.nn.dropout(A, keep_prob=keep_prob)

    return logits, Ad, update_ema


def def_conv_layer_bn(input, filter, offset, strides, is_training, iter, keep_prob, padding='SAME'):
    # no 'scale' (alpha) param if relu, no biases too, because we have scale
    ema = tf.train.ExponentialMovingAverage(wavg_decay, iter)

    conv = tf.nn.conv2d(input, filter=filter, strides=strides, padding=padding)

    # We have to handle the convolution case where the stats are computed not just across the minibatch but also
    # across all the x,y positions a patch can take. That is what tf.nn.moments(Ylogits, [0, 1, 2]) does. It
    # computes the stats across tensor directions 0 (image instances in mini-batch), 1 (x positions)
    # and 2 (y positions) - per channel.

    batch_mean, batch_var = tf.nn.moments(conv, [0, 1, 2])
    update_ema = ema.apply([batch_mean, batch_var])
    m = tf.cond(is_training, lambda: batch_mean, lambda: ema.average(batch_mean))
    v = tf.cond(is_training, lambda: batch_var, lambda: ema.average(batch_var))

    bn = tf.nn.batch_normalization(conv, m, v, offset, None, variance_epsilon=1e-8)
    A = tf.nn.relu(bn)
    Ad = tf.nn.dropout(A, keep_prob=keep_prob)

    return Ad, update_ema

##### placeholders #####

# is this training?
is_training = tf.placeholder(tf.bool)
# weighted average decay rate
wavg_decay = tf.placeholder(tf.float32)
# iteration tracking for EMA
iter = tf.placeholder(tf.int32)
# global step for learning rate
global_step = tf.Variable(0, trainable=False)
# dropout
keep_prob = tf.placeholder(tf.float32)
# learning rate
alpha = tf.Variable(0.0, trainable=False)
# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# labels
Y_ = tf.placeholder(tf.float32, [None, 10])


##### init layers #####
W_conv1, b1 = init_conv_layer([5, 5, 1, 6])
W_conv2, b2 = init_conv_layer([4, 4, 6, 12])
W_conv3, b3 = init_conv_layer([4, 4, 12, 24])
W4, offset4 = init_fc_layer([7 * 7 * 24, 200])
W5, offset5 = init_fc_layer([200, 10])


##### define model #####
A1, ema1 = def_conv_layer_bn(X, W_conv1, b1, strides=[1, 1, 1, 1],
                             is_training=is_training, iter=iter, padding='SAME', keep_prob=keep_prob)
A2, ema2 = def_conv_layer_bn(A1, W_conv2, b2, strides=[1, 2, 2, 1], is_training=is_training, iter=iter, padding='SAME'
                             , keep_prob=keep_prob)
A3, ema3 = def_conv_layer_bn(A2, W_conv3, b3, strides=[1, 2, 2, 1], is_training=is_training, iter=iter, padding='SAME'
                             , keep_prob=keep_prob)

A3_flat = tf.reshape(A3, shape=[-1, 7 * 7 * 24])

_, A4, ema4 = def_fc_layer_bn(A3_flat, W4, offset4, is_training, iter, keep_prob=keep_prob)

Ylogits, _ = def_fc_layer(A4, W5, offset5, keep_prob=1.0)

Y = tf.nn.softmax(Ylogits)

ema_updates = tf.group(ema1, ema2, ema3, ema4)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
# cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
#                                                           # *10 because  "mean" included an unwanted division by 10

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# learning rate
lr = tf.train.exponential_decay(0.02, global_step, 1000, 0.95)
alpha = tf.assign(alpha, lr)
# training, learning rate = 0.005
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy, global_step=global_step)

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W_conv1, [-1]),
                        tf.reshape(W_conv2, [-1]),
                        tf.reshape(W_conv3, [-1]),
                        tf.reshape(W4, [-1]),
                        tf.reshape(W5, [-1])], 0)

allbiases  = tf.concat([tf.reshape(b1, [-1]),
                        tf.reshape(b2, [-1]),
                        tf.reshape(b3, [-1]),
                        tf.reshape(offset4, [-1]),
                        tf.reshape(offset5, [-1])], 0)

I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # learning rate decay
    lrmax = 0.003
    lrmin = 0.0001

    learning_rate = lrmin + (lrmax - lrmin) * exp(-i / 2000)

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b, lr  = sess.run([accuracy, cross_entropy, I, allweights, allbiases, alpha],
                                  feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 0.75, is_training: True,
                                             wavg_decay: 0.999})

        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " learning rate: " + str(lr))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0,
                                       is_training: False, wavg_decay: 0.999})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run([train_step, ema_updates], feed_dict={X: batch_X, Y_: batch_Y, alpha: learning_rate, keep_prob: 0.75, is_training: True,
                                             wavg_decay: 0.999, iter: i})


#import datetime as dt

#start = dt.datetime.now()
datavis.animate(training_step, iterations=10000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)
# for i in range(2000):
#     training_step(i, False, False)
# end = dt.datetime.now()
# print('exec time:', end-start)
# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
