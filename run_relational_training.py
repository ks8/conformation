""" Run relational network training. """
from conformation.run_relational_training import run_relational_training, Args
from conformation.create_logger import create_logger

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    run_relational_training(args, logger)
