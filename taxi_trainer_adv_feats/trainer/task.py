from trainer.model import train_and_evaluate
import shutil
import click

@click.command()
@click.option('--out-dir', default='/tmp/taxi_trainer')
@click.option('--resume', is_flag=True, help='do not cleanup out_dir before running; continue training.')
@click.option('--train-path', required=True)
@click.option('--eval-path', required=True)
@click.option('--max-steps', default=2000)
@click.option('--nbuckets', required=True, type=int)
@click.option('--job-dir', default=None)
@click.argument('hidden-units', nargs=-1, type=int)
def main(out_dir, train_path, eval_path, hidden_units, max_steps, nbuckets, resume, job_dir):
    if not resume:
        shutil.rmtree(out_dir, ignore_errors=True)
    metrics = train_and_evaluate(out_dir, train_path, eval_path, hidden_units, nbuckets, max_steps)

    # in training on GCP, metrics are undefined
    if metrics is not None:
        print(metrics[0])


if __name__ == '__main__':
    main()
