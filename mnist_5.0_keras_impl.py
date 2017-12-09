# encoding: UTF-8


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

from keras_tqdm import TQDMCallback
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from datetime import datetime as dt
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# define model

dropout_rate = 0.2

model = Sequential()

model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))

model.add(Conv2D(12, kernel_size=(4, 4), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))

model.add(Conv2D(24, kernel_size=(4, 4), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))

model.add(Flatten())

model.add(Dense(200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.3, decay=0.01),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

start = dt.now()
fBestModel = 'best_model.h5'
fModelStruct= 'model_struct.json'
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

model.fit(x=mnist.train.images, y=mnist.train.labels,
          validation_data=(mnist.test.images, mnist.test.labels),
          batch_size=128, epochs=1, verbose=True,
          callbacks=[TensorBoard(log_dir="/tmp/mnist/{}/{}".format(str(dt.now().date()), str(dt.now().time())),
                                 histogram_freq=1,
                                 write_images=True
                                 ),
                     best_model
                     ])
metrics = model.evaluate(mnist.test.images, mnist.test.labels)
print('eval:', list(zip(model.metrics_names, metrics)))
end = dt.now()
print('exec time:', end-start)

with open(fModelStruct, 'w') as f:
    f.write(model.to_json())

print('model structure saved to \'{}\' and data saved to: \'{}\''.format(fModelStruct, fBestModel))




