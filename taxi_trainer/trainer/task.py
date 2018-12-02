from trainer.model import task
import shutil
import click

@click.command()
@click.option('--out-dir', default='/tmp/taxi_trainer')
@click.option('--resume', is_flag=True, help='do not cleanup out_dir before running; continue training.')
@click.option('--train-path', required=True, type=click.Path(exists=True))
@click.option('--eval-path', required=True, type=click.Path(exists=True))
@click.option('--max-steps', default=2000)
@click.argument('hidden-units', nargs=-1, type=int)
def main(out_dir, train_path, eval_path, hidden_units, max_steps, resume):
    if not resume:
        shutil.rmtree(out_dir, ignore_errors=True)
    metrics, _ = task(out_dir, train_path, eval_path, hidden_units, max_steps)
    print(metrics)


if __name__ == '__main__':
    main()
