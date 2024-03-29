""" Run basic NF sampling. """
import os

from conformation.create_logger import create_logger
from conformation.run_basic_nf_sampling import run_basic_nf_sampling, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    run_basic_nf_sampling(args, logger)
