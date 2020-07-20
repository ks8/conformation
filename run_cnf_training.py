""" Run neural network training. """
from conformation.run_cnf_training import run_training
from conformation.train_args import Args
from conformation.create_logger import create_logger

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    run_training(args, logger)
