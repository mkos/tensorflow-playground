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

## Running cloud training_path

Create bucket
```
$ gsutil mb -l $REGION $BUCKET
```

Copy data
```
gsutil cp data/taxi_adv_feats/* $BUCKET
```

Run distributed job
```
$ gcloud ml-engine jobs submit training ${JOBNAME}_6 \
    --module-name trainer.task \
    --package-path $PWD/trainer \
    --job-dir $BUCKET/output \
    --region europe-west1 --scale-tier=BASIC --runtime-version=1.9 \
    -- \
    --train-path $BUCKET/train.csv \
    --eval-path $BUCKET/valid.csv \
    --max-steps 20000 \
    --nbuckets 10 --out-dir $BUCKET/output/taxi_trainer \
    100 50 20
```

Run distributed hyperparam training job:

```
$ gcloud ml-engine jobs submit training taxi_job_0 \
    --package-path=$PWD/trainer \
    --module-name=trainer.task \
    --job-dir=$BUCKET/output/ \
    --runtime-version=1.9 \
    --config=$PWD/hyperparam.yaml \
    --region=europe-west1
    -- \
    --train-path=$BUCKET/data/train.csv \
    --eval-path=$BUCKET/data/valid.csv \
    --out-dir=$BUCKET/output/taxi_trainer \
    --max-steps=5000
```

view job logs

```
$ gcloud ml-engine jobs stream-logs ${JOBNAME}_6
```


## Troubleshooting

If you get error:
[gcloud ml-engine local predict RuntimeError: Bad magic number in .pyc file](https://stackoverflow.com/questions/48824381/gcloud-ml-engine-local-predict-runtimeerror-bad-magic-number-in-pyc-file)

do the following:
1. Use `--verbosity=debug` to determine where is your google-cloud-sdk installation
2. Go to that folder and issue `find . -name "*.pyc" | xargs rm`
3. run command again.
