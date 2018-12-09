import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import fire
import shutil

"""
Run this with:
$ python house_value_1.0_keras.py --num-training-steps 100 --model-dir=/tmp/housing_trained
"""

tf.logging.set_verbosity(tf.logging.INFO)
DATA_PATH = "https://storage.googleapis.com/ml_universities/california_housing_train.csv"
OUTDIR = './housing_trained'
COLUMNS = 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house,longitude,latitude'.split(',')
LABEL = "median_house_value"
KERAS_PREFIX = 'keras_'

def make_datasets(path):
    df = pd.read_csv(path, sep=",")

    np.random.seed(seed=1) #makes split reproducible
    msk = np.random.rand(len(df)) < 0.8

    df['num_rooms'] = df['total_rooms'] / df['households']
    df['num_bedrooms'] = df['total_bedrooms'] / df['households']
    df['persons_per_house'] = df['population'] / df['households']
    df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], axis=1, inplace=True)

    traindf = df[msk]
    evaldf = df[~msk]

    return traindf, evaldf


# this part is not used right now in any way

# def make_feat_cols():
#     featcols = [
#         tf.feature_column.numeric_column(col) for col
#         in 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house'.split(',')
#     ]
#     featcols.append(
#         tf.feature_column.bucketized_column(
#             tf.feature_column.numeric_column('latitude'),
#             boundaries=np.linspace(32.5, 42, 10).tolist()
#         ))
#     featcols.append(
#         tf.feature_column.bucketized_column(
#             tf.feature_column.numeric_column('longitude'),
#             boundaries=np.linspace(-124.3, -114.3, 5).tolist()
#         ))
#
#     return featcols


def make_keras_estimator(hidden_units):

    input_layers = [keras.layers.Input(shape=(1,), name=f) for f in COLUMNS]
    X = keras.layers.concatenate(input_layers)
    X = keras.layers.Reshape((-1, 1))(X)

    for units_num in hidden_units:
        X = keras.layers.Dense(units_num)(X)
        X = keras.layers.Activation('relu')(X)

    X = keras.layers.Dense(1, name='predictions')(X) # < 'name' is required for custom metrics

    model = keras.models.Model(inputs=input_layers, outputs=X)
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mae']
    )

    return keras.estimator.model_to_estimator(model)


def scaled_rmse(scale):
    def _rmse(labels, predictions):
        print('>>>>>>>>>>', predictions)
        pred_values = tf.cast(predictions['predictions'], tf.float64)
        # return to original scale in metric
        return {'rmse': tf.metrics.root_mean_squared_error(labels * scale, pred_values * scale)}

    return _rmse


def task(model_dir, num_training_steps, resume=False, scale=100000, learning_rate=0.1, batch_size=512,
         dropout=0.1, hidden_units='100,50,20'):

    if not resume:
        shutil.rmtree(model_dir, ignore_errors=True)

    traindf, evaldf = make_datasets(DATA_PATH)

    train_spec = tf.estimator.TrainSpec(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=traindf[COLUMNS],
            y=traindf[LABEL] / scale, # scale down
            num_epochs=None,
            batch_size=batch_size,
            shuffle=True,
        ),
        max_steps=num_training_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=evaldf[COLUMNS],
            y=evaldf[LABEL] / scale, # scale down
            num_epochs=1,
            batch_size=len(evaldf),
            shuffle=False
        ),
        steps=None,
        start_delay_secs=1,
        throttle_secs=10,
    )

    estimator = make_keras_estimator(hidden_units.split(','))
    #estimator = tf.contrib.estimator.add_metrics(estimator, scaled_rmse(scale))  # for now this doesn't work okay

    metrics, _ = tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )
    return metrics


if __name__ == '__main__':
    fire.Fire(task)
