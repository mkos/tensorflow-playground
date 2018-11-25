#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
print('tensorflow version:', tf.__version__)

graph = tf.Graph()

with graph.as_default() as g:

    x = tf.placeholder(tf.float32, shape=())
    params = tf.placeholder(tf.float32, shape=(5,))
    delta = tf.placeholder(tf.float32, shape=())
    max_iters = tf.placeholder(tf.int64, shape=())

    def loop_body(x_prev, x):
        pows = tf.math.pow(tf.tile(tf.expand_dims(x, axis=0), [5]), [0, 1, 2, 3, 4])

        f = tf.reduce_sum(params * pows)
        f_prim = tf.reduce_sum([1.0, 2.0, 3.0, 4.0] * params[1:] * pows[:-1])
        f_bis = tf.reduce_sum([2.0, 6.0, 12.0] * params[2:] * pows[:-2])

        update = (2.0 * f * f_prim) / (2.0 * tf.math.pow(f_prim, 2) - f * f_bis)

        x_next = x - update

        return x, x_next

    def cond(x, x_next):
        return tf.abs(x - x_next) > delta

    _, approx = tf.while_loop(cond, loop_body, (500.0, x), maximum_iterations=max_iters)

with tf.Session(graph=graph) as sess:

    approximate_root = sess.run(approx,
        feed_dict={
            x: 0.0,
            params: [10.0, 8.0, 6.0, 4.0, 2.0], #[6.0, 5.0, 4.0, 3.0, 2.0],
            delta: 0.001,
            max_iters: 1000000
        })

    print('approximate root:', approximate_root)
