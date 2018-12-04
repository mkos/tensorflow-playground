Predicting NYC taxi fares. Based on Coursera's Google ML specialisation.

This version has improved and engineered features.
## Training locally

```
gcloud ml-engine local train \
        --module-name trainer.task \
        --package-path ./trainer \
        -- \
        --train-path ../data/taxi_adv_feats/train.csv \
        --eval-path ../data/taxi_adv_feats/valid.csv \
        --nbuckets 16 \
        64 64 64 8
```

## Running local prediction

Change the number of the model in the latest_exporter below:

```
gcloud ml-engine local predict \
    --model-dir=/tmp/taxi_trainer/export/latest_exporter/1543696421
    --json-instances=test.json
```

## Troubleshooting

If you get error:
[gcloud ml-engine local predict RuntimeError: Bad magic number in .pyc file](https://stackoverflow.com/questions/48824381/gcloud-ml-engine-local-predict-runtimeerror-bad-magic-number-in-pyc-file)

do the following:
1. Use `--verbosity=debug` to determine where is your google-cloud-sdk installation
2. Go to that folder and issue `find . -name "*.pyc" | xargs rm`
3. run command again.
