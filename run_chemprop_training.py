""" Run chem prop network training. """
from conformation.create_logger import create_logger
from conformation.run_chemprop_training import run_chemprop_training, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    run_chemprop_training(args, logger)
