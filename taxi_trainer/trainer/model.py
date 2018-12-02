import tensorflow as tf

CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0] # fare_amount
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]


def make_feature_cols():
    return [tf.feature_column.numeric_column(col) for col in FEATURES]


def serving_input_receiver_fn():
    feature_placeholders = {col: tf.placeholder(tf.float32, shape=None) for col in CSV_COLUMNS}
    features = feature_placeholders  # no feature engineering

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def make_dataset(path, mode, batch_size=512):
    def _input_fn():

        def csv_decode(line):
            line_tensors = tf.decode_csv(line, DEFAULTS)
            features = dict(zip(CSV_COLUMNS, line_tensors))
            label = features.pop(LABEL_COLUMN)
            return features, label # TODO

        dataset = \
        (tf.data.Dataset
        .list_files(path)
        .flat_map(tf.data.TextLineDataset)
        .map(csv_decode)
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # run indefinitely
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1

        return (
            dataset
            .repeat(num_epochs)
            .batch(batch_size)
            .make_one_shot_iterator()
            .get_next()
        )

    return _input_fn


def task(out_dir, training_path, eval_path, hidden_units, max_steps=None):

    train_spec = tf.estimator.TrainSpec(
        make_dataset(training_path, mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=max_steps
    )

    exporter = tf.estimator.LatestExporter(
        'latest_exporter',
        serving_input_receiver_fn=serving_input_receiver_fn  # notice it's passing function, not it's output
        )
    
    eval_spec = tf.estimator.EvalSpec(
        make_dataset(eval_path, mode=tf.estimator.ModeKeys.EVAL),
        exporters=exporter
    )

    model = tf.estimator.DNNRegressor(
        feature_columns=make_feature_cols(),
        hidden_units=hidden_units,
        model_dir=out_dir
        )

    metrics = tf.estimator.train_and_evaluate(
        estimator=model,
        train_spec=train_spec,
        eval_spec=eval_spec,
    )

    return metrics
