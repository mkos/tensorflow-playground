from trainer.model import train_and_evaluate
import shutil
import click
import os
import json

@click.command()
@click.option('--out-dir', default='/tmp/taxi_trainer')
@click.option('--resume', is_flag=True, help='do not cleanup out_dir before running; continue training.')
@click.option('--train-path', required=True)
@click.option('--eval-path', required=True)
@click.option('--max-steps', default=2000)
@click.option('--nbuckets', default=10, type=int)
@click.option('--batch-size', default=512, type=int)
@click.option('--learning-rate', default=0.01, type=float)
@click.option('--job-dir', default=None)
@click.option('--hidden-units', default="8 4")
def main(out_dir, train_path, eval_path, hidden_units, max_steps, nbuckets, resume, job_dir,
         batch_size, learning_rate):

    # do not overwrite results when tuning hyperparams
    out_dir = os.path.join(
        out_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )

    # cleanup output dir if not resuming
    if not resume:
        shutil.rmtree(out_dir, ignore_errors=True)

    hidden_units = hidden_units.split(' ')

    metrics = train_and_evaluate(out_dir, train_path, eval_path, hidden_units, nbuckets, max_steps,
                                 batch_size, learning_rate
                                )

    # in training on GCP, metrics are undefined
    if metrics is not None:
        print(metrics[0])


if __name__ == '__main__':
    main()
