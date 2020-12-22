""" Run basic NF training. """
import os

from conformation.run_basic_nf_training import run_basic_nf_training
from conformation.train_args import Args
from conformation.create_logger import create_logger

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    run_basic_nf_training(args, logger)
