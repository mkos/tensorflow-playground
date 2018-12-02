# Tensorflow playground

This repo originally started as a fork of Martin's Gorners excellent tutorial repo
[Tensorflow and deep learning, without a PhD](https://github.com/martin-gorner/tensorflow-mnist-tutorial), which quickly
became my sandbox for experimentation with new features of tensor flow as well as new concepts from deep learning.

Since some of my files were partial copies of Martin's files, where I have added new elements while following the tutorial.
I kept the original repo in `originals` directory with all the files from original repo (I just updated few imports so everyting
was working as expected).

Following files are following [codelab](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) based on the
mentioned eariler tutorial:
* `mnist_2.0_five_layers_sigmoid.py`
* `mnist_2.1_five_layers_relu_lrdecay.py`
* `mnist_2.2_five_layers_relu_lrdecay_dropout.py`
* `mnist_3.0_convolutional.py`
* `mnist_3.1_convolutional_moar_filters.py`

Remaining files are modifications of `mnist_3.1_convolutional_moar_filters.py` with less and less dependencies on original
project files.

* `mnist_4.0_convolutional_moar_filters_batchnorm.py` - added batch normalization
* `mnist_4.1_convolutional_moar_filters_batchnorm_tensorboard.py` - instead of using custom visualizer, go with tensorboard
for simple (scalar) metrics
* `mnist_4.2_conv_bn_tensorboard_with_weights.py` - added plotting layer's weights, biases, activations and images to tensorboard.
Based on an excellent video [Hands-on TensorBoard (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4&index=5&list=PLOU2XLYxmsIKGc_NBoIhTn2Qhraji53cv)
(+accompanying [repo](https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial))
* `mnist_5.0_keras_impl.py` - reimplementation in Keras
* `mnist_6.0_canned_estimator.py` - reimplementation using Estimator API (simple canned estimator, [source](https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-estimators))
* `mnist_6.1_custom_estimator.py` - using custom estimators

## Other

* `halleys.py` - calculating roots of a polynomial using Halley's method (based on exercise from Intro to Tensorflow Coursera course)
* `nyc_taxi_1.0_pandas_input.py` - nyc taxifare dataset, pandas input, canned estimator
* `taxi_trainer` - project with estimators, ready to use with CMLE, canned estimator, input functions, serving, etc.
