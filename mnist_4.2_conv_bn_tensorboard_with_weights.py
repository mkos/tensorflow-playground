# encoding: UTF-8

# added: tensorboard plotting of (train|test) accuracy and cross entropy

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from math import exp
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

##### helper functions #####

def def_fc_layer_logits(input, shape):
    with tf.name_scope('output'):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        b = tf.Variable(tf.ones([shape[1]]) / 10)

        logits = tf.matmul(input, W) + b


    return logits

def def_fc_layer_bn(input, shape, is_training, iter, keep_prob, name='fc'):
    with tf.name_scope(name):
        # init vars
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        offset = tf.Variable(tf.ones([shape[1]]) / 10)

        # no 'scale' (alpha) param if relu, no biases too, because we have scale
        ema = tf.train.ExponentialMovingAverage(wavg_decay, iter)

        logits = tf.matmul(input, W)
        batch_mean, batch_var = tf.nn.moments(logits, [0])
        update_ema = ema.apply([batch_mean, batch_var])

        m = tf.cond(is_training, lambda: batch_mean, lambda: ema.average(batch_mean))
        v = tf.cond(is_training, lambda: batch_var, lambda: ema.average(batch_var))

        bn = tf.nn.batch_normalization(logits, m, v, offset, None, variance_epsilon=1e-8)
        A = tf.nn.relu(bn)
        Ad = tf.nn.dropout(A, keep_prob=keep_prob)

        return Ad, update_ema


def def_conv_layer_bn(input, shape, strides, is_training, iter, keep_prob, padding='SAME', name='conv'):
    with tf.name_scope(name):
        # init vars
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        offset = tf.Variable(tf.ones([shape[3]])/10)

        # no 'scale' (alpha) param if relu, no biases too, because we have scale
        ema = tf.train.ExponentialMovingAverage(wavg_decay, iter)

        conv = tf.nn.conv2d(input, filter=W, strides=[1, strides, strides, 1], padding=padding)

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

        s_w = tf.summary.histogram('weights', W)
        s_b = tf.summary.histogram('offsets', offset)
        s_a = tf.summary.histogram('activations', A)

        summ = tf.summary.merge([s_w, s_b, s_a])

        return Ad, update_ema, summ

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


##### define model #####
A1, ema1, summ_1 = def_conv_layer_bn(X, shape=[5, 5, 1, 6], strides=1, is_training=is_training,
                             iter=iter, padding='SAME', keep_prob=keep_prob, name='conv_1')
A2, ema2, summ_2 = def_conv_layer_bn(A1, shape=[4, 4, 6, 12], strides=2, is_training=is_training,
                             iter=iter, padding='SAME', keep_prob=keep_prob, name='conv_2')
A3, ema3, summ_3 = def_conv_layer_bn(A2, shape=[4, 4, 12, 24], strides=2, is_training=is_training,
                             iter=iter, padding='SAME', keep_prob=keep_prob, name='conv_3')

A3_flat = tf.reshape(A3, shape=[-1, 7 * 7 * 24])

A4, ema4 = def_fc_layer_bn(A3_flat, shape=[7 * 7 * 24, 200], is_training=is_training,
                              iter=iter, keep_prob=keep_prob, name='fc1')

Ylogits = def_fc_layer_logits(A4, shape=[200, 10])

Y = tf.nn.softmax(Ylogits)

ema_updates = tf.group(ema1, ema2, ema3, ema4)

# softmax activation with cross-entropy, normalized for batches of 100 images
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)) * 100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# learning rate
lr = tf.train.exponential_decay(0.02, global_step, 1000, 0.95)
alpha = tf.assign(alpha, lr)
# training, learning rate = 0.005
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy, global_step=global_step)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data, update_summary):

    # learning rate decay
    lrmax = 0.003
    lrmin = 0.0001

    learning_rate = lrmin + (lrmax - lrmin) * exp(-i / 2000)

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    if update_summary:
        # s = sess.run(merged_summary, feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 0.75, is_training: True,
        #                                      wavg_decay: 0.999})
        # writer.add_summary(s, i)

        pass

    # compute training values for visualisation
    if update_train_data:
        a, c, lr, acc_s, xent_s, hist_s  = sess.run([accuracy, cross_entropy, alpha, training_accuracy, cross_entropy_summary,
                                             summ_hist],
                                  feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 0.75, is_training: True,
                                             wavg_decay: 0.999})

        writer.add_summary(acc_s, i)
        writer.add_summary(xent_s, i)
        writer.add_summary(hist_s, i)

        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " learning rate: " + str(lr))

    # compute test values for visualisation
    if update_test_data:
        a, c, t_acc_s = sess.run([accuracy, cross_entropy, test_accuracy],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0,
                                       is_training: False, wavg_decay: 0.999})

        writer.add_summary(t_acc_s, i)

        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run([train_step, ema_updates], feed_dict={X: batch_X, Y_: batch_Y, alpha: learning_rate, keep_prob: 0.75, is_training: True,
                                             wavg_decay: 0.999, iter: i})


##### main loop #####
from datetime import datetime as dt
cross_entropy_summary = tf.summary.scalar('cross entropy', cross_entropy)
training_accuracy = tf.summary.scalar('training accuracy', accuracy)
test_accuracy = tf.summary.scalar('test accuracy', accuracy)
summ_hist = tf.summary.merge([summ_1, summ_2, summ_3])

# save for tensorboard


writer = tf.summary.FileWriter("/tmp/mnist/{}/{}".format(str(dt.now().date()), str(dt.now().time())
), flush_secs=10)
writer.add_graph(sess.graph)


start = dt.now()
for i in range(5000):

    if i % 5 == 0:
        training_step(i, False, False, True)
        if i % 10 == 0:
            training_step(i, False, True, False)
            if i % 50 == 0:
                training_step(i, True, False, False)
    else:
        training_step(i, False, False, False)

end = dt.now()
print('exec time:', end-start)

