""" Run basic Metropolis-Hastings sampling. """
import os

from conformation.create_logger import create_logger
from conformation.basic_metropolis import basic_metropolis, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    basic_metropolis(args, logger)
