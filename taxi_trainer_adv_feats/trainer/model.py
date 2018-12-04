import tensorflow as tf
import numpy as np

CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]
KEY_FEATURE_COLUMN = 'key'


RAW_INPUT_COLS = [
    # New features
    tf.feature_column.categorical_column_with_vocabulary_list(
        'dayofweek',
        vocabulary_list='Sun,Mon,Tues,Wed,Thu,Fri,Sat'.split(',')
    ),
    tf.feature_column.categorical_column_with_identity('hourofday', num_buckets=24),

    # numerical features
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('passengers'),

    # engineered features in add_engineered()
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean'),
]


def add_engineered(features):
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']

    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    features['euclidean'] = tf.sqrt(latdiff * latdiff + londiff * londiff)

    return features


def serving_input_receiver_fn():
    feature_placeholders = {col.name: tf.placeholder(tf.float32, shape=None)
                            for col in RAW_INPUT_COLS[2:]}
    feature_placeholders['dayofweek'] = tf.placeholder(tf.string, shape=None)
    feature_placeholders['hourofday'] = tf.placeholder(tf.int32, shape=None)
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(add_engineered(features), feature_placeholders)


def make_dataset(path, mode, batch_size=512):
    def _input_fn():

        def csv_decode(line):
            line_tensors = tf.decode_csv(line, DEFAULTS)
            features = dict(zip(CSV_COLUMNS, line_tensors))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label

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


def build_estimator(model_dir, nbuckets, hidden_units):

    (dayofweek, hourofday, plon, plat, dlon, dlat, pcount, latdiff, londiff, euclidean) = RAW_INPUT_COLS

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()

    # bucketize/OHE
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    # feature cross
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)

    wide_columns = [
        # feature crosses
        ploc, dloc, pd_pair, day_hr,
        # sparse cols - OHEd
        dayofweek, hourofday,
        # linear relationship
        pcount
    ]

    deep_columns = [
        # embeddings: pd_pair, day_hr, 10
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        # numeric columns
        plat, dlat, plon, dlon,
        latdiff, londiff, euclidean
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        model_dir=model_dir
        )

    estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator


def add_eval_metrics(labels, predictions):
    return {'rmse': tf.metrics.root_mean_squared_error(labels, predictions['predictions'])}


def train_and_evaluate(out_dir, training_path, eval_path, hidden_units, nbuckets, max_steps=None):

    tf.logging.set_verbosity(tf.logging.INFO)
    
    estimator = build_estimator(out_dir, nbuckets, hidden_units)

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

    metrics = tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec,
    )

    return metrics
