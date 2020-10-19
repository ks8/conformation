""" General framework for parallel tempering MCMC. """
import os

from conformation.create_logger import create_logger
from conformation.parallel_tempering import parallel_tempering, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    parallel_tempering(args, logger)
