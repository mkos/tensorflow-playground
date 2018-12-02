import tensorflow as tf

# In CSV, label is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0] # fare_amount


def make_feature_cols():
    return [tf.feature_columns.numeric_column(feat)
        for feat in FEATURES if feat != LABEL
    ]

estimator = tf.estimator.DNNRegressor(
    feature_columns=make_feature_cols()
    hidden_units=[32, 8, 2]
)

estimator.train
