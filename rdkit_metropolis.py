""" Metropolis-Hastings conformational search using RDKit. """
import os

from conformation.create_logger import create_logger
from conformation.rdkit_metropolis import rdkit_metropolis, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    rdkit_metropolis(args, logger)
