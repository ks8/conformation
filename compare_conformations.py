""" Compare two sets of conformations. """
import os

from conformation.create_logger import create_logger
from conformation.compare_conformations import compare_conformations, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    compare_conformations(args, logger)
