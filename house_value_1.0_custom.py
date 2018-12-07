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
COLUMNS = 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house,longitude,latitude'.split(',')
LABEL = "median_house_value"

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
    print(df.columns)

    return traindf, evaldf


def make_feat_cols():
    featcols = [
        tf.feature_column.numeric_column(col) for col
        in 'housing_median_age,median_income,num_rooms,num_bedrooms,persons_per_house'.split(',')
    ]
    featcols.append(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('latitude'),
            boundaries=np.linspace(32.5, 42, 10).tolist()
        ))
    featcols.append(
        tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column('longitude'),
            boundaries=np.linspace(-124.3, -114.3, 5).tolist()
        ))

    return featcols


def custom_estimator(features, labels, mode, params):
    # 0. Extract data from feature columns
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    # 1. Define Model Architecture
    predictions = tf.layers.dense(input_layer, 1)
    # 2. Loss function, training/eval ops
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
        loss = tf.losses.mean_squared_error(labels, predictions)
        optimizer = tf.train.FtrlOptimizer(learning_rate=params['learning_rate'])
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step= tf.train.get_global_step(),
            optimizer=optimizer,
            learning_rate=params['learning_rate']
        )
        eval_metric_ops = {'rmse': tf.metrics.root_mean_squared_error(
            labels * params['scale'], predictions * params['scale'])}
    else:
        train_op = None
        loss = None
        eval_metric_ops=None

    # 3. Create predictions
    predictions_dict = {'predicted': predictions}

    # 4. Create export outputs
    export_outputs = {"regression_export_outputs": tf.estimator.export.RegressionOutput(value = predictions)}

    # 5. Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        export_outputs=export_outputs,
        eval_metric_ops=eval_metric_ops
    )


def task(model_dir, num_training_steps, resume=False, scale=100000, learning_rate=0.1, batch_size=512,
         dropout=0.1, hidden_units='100,50,20'):

    if not resume:
        shutil.rmtree(model_dir, ignore_errors=True)

    traindf, evaldf = make_datasets(DATA_PATH)

    estimator = tf.estimator.Estimator(
        model_fn=custom_estimator,
        model_dir=model_dir,
        params={
            'feature_columns': make_feat_cols(),
            'hidden_units': hidden_units.split(','),
            'learning_rate': learning_rate,
            'scale': scale,
        }
    )


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

    metrics, _ = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return metrics


if __name__ == '__main__':
    fire.Fire(task)
