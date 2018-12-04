import tensorflow as tf
import pandas as pd
import numpy as np
import fire
import shutil

"""
Run this with:
$ python house_value_1.0_basic.py --num-training-steps 2000 --model-dir=/tmp/housing_trained [--resume]
"""

tf.logging.set_verbosity(tf.logging.INFO)
DATA_PATH = "https://storage.googleapis.com/ml_universities/california_housing_train.csv"
OUTDIR = './housing_trained'
COLUMNS = ["num_rooms"]
LABEL = "median_house_value"

def make_datasets(path):
    df = pd.read_csv(path, sep=",")

    np.random.seed(seed=1) #makes split reproducible
    msk = np.random.rand(len(df)) < 0.8

    df['num_rooms'] = df['total_rooms'] / df['households']

    traindf = df[msk]
    evaldf = df[~msk]

    return traindf, evaldf


def make_feat_cols():
    return [
        tf.feature_column.numeric_column('num_rooms')
    ]


def scaled_rmse(scale):
    def _rmse(labels, predictions):
        pred_values = tf.cast(predictions['predictions'], tf.float64)
        # return to original scale in metric
        return {'rmse': tf.metrics.root_mean_squared_error(labels * scale, pred_values * scale)}

    return _rmse


def task(num_training_steps, model_dir, resume=False, scale=100000):

    if not resume:
        shutil.rmtree(model_dir, ignore_errors=True)

    traindf, evaldf = make_datasets(DATA_PATH)

    estimator = tf.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=make_feat_cols()
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, scaled_rmse(scale))

    train_spec = tf.estimator.TrainSpec(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=traindf[COLUMNS],
            y=traindf[LABEL] / scale, # scale down
            num_epochs=None,
            shuffle=True,
        ),
        max_steps=num_training_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=evaldf[COLUMNS],
            y=evaldf[LABEL] / scale, # scale down
            num_epochs=1,
            shuffle=False
        ),
        steps=None,
        start_delay_secs=1,
        throttle_secs=10,
    )

    metrics, _ = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return metrics


if __name__ == '__main__':
    fire.Fire(task)
