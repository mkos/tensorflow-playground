# encoding: UTF-8
# source: https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-estimators

import numpy as np
import tensorflow as tf
from datetime import datetime as dt
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

image_size = 28
labels_size = 10
hidden_size = 1024

# Read in the MNIST dataset
mnist = input_data.read_data_sets("data", one_hot=True)

def input_fn(dataset):
    features = dataset.images
    labels = dataset.labels.astype(np.int32)
    return features, labels

# Define the Estimator
#feature_columns = [tf.contrib.layers.real_valued_column("", dimension=image_size*image_size)]

def model_fn(features, labels, mode, params):

    X_flat = tf.reshape(features, shape=[-1, 784])

    # W1 = tf.get_variable('W1', [784, 200], initializer=tf.random_normal_initializer(stddev=0.1))
    W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
    # b1 = tf.get_variable('b1', [200], initializer=tf.zeros_initializer)
    b1 = tf.Variable(tf.ones([200]) / 10)
    # W2 = tf.get_variable('W2', [200, 100], initializer=tf.random_normal_initializer(stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
    # b2 = tf.get_variable('b2', [100], initializer=tf.zeros_initializer)
    b2 = tf.Variable(tf.ones([100]) / 10)
    # W3 = tf.get_variable('W3', [100, 60], initializer=tf.random_normal_initializer(stddev=0.1))
    W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
    # b3 = tf.get_variable('b3', [60], initializer=tf.zeros_initializer)
    b3 = tf.Variable(tf.ones([60]) / 10)
    # W4 = tf.get_variable('W4', [60, 30], initializer=tf.random_normal_initializer(stddev=0.1))
    W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
    # b4 = tf.get_variable('b4', [30], initializer=tf.zeros_initializer)
    b4 = tf.Variable(tf.ones([30]) / 10)
    # W5 = tf.get_variable('W5', [30, 10], initializer=tf.random_normal_initializer(stddev=0.1))
    W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
    # b5 = tf.get_variable('b5', [10], initializer=tf.zeros_initializer)
    b5 = tf.Variable(tf.ones([10]) / 10)

    a1 = tf.nn.relu(tf.matmul(X_flat, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    a3 = tf.nn.relu(tf.matmul(a2, W3) + b3)
    a4 = tf.nn.relu(tf.matmul(a3, W4) + b4)
    Ylogits = tf.matmul(a4, W5) + b5
    Y = tf.nn.softmax(Ylogits)


    cross_entropy = tf.losses.softmax_cross_entropy(labels, Ylogits)
    tf.summary.scalar('xent', cross_entropy)


    train_op = tf.train.AdamOptimizer(0.003).minimize(cross_entropy, tf.train.get_global_step())

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, Y)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

print('building model...')
classifier = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir="/tmp/mnist/{}/{}".format(str(dt.now().date()), str(dt.now().time())),
                                    config=tf.estimator.RunConfig().replace(save_summary_steps=1, log_step_count_steps=1)
                                    )

class PrinterHook(tf.train.SessionRunHook):
    def __init__(self):
        self.counter = 1
    def after_create_session(self, session, coord):
        print('session created.')

    def begin(self):
        print('begin using default graph in the session')

    def after_run(self, run_context, run_values):
        print('step: ', self.counter)
        self.counter += 1


print('training model...')
# Fit the model
classifier.train(input_fn=lambda: input_fn(mnist.train), steps=10,
                 hooks=[PrinterHook(),
                        #tf.train.LoggingTensorHook({'xent': 'xent'}, every_n_iter=1)
                        ])

# Evaluate the model on the test data
test_accuracy = classifier.evaluate(input_fn=lambda: input_fn(mnist.test), steps=1)

print("\nTest accuracy: {}".format(test_accuracy["accuracy"]))

# Predict the new examples and compare with the onderlying values
# features = mnist.validation.images[:10]
# labels = mnist.validation.labels[:10].astype(np.int32)
# predictions = list(classifier.predict(x=features))

# print("\nPredicted labels from validation set: %s"%predictions)
# print("Underlying values: %s"%list(labels))

# TODO: predicting, better input fn with tf.estimator.inputs.numpy_input_fn