""" Systematic conformer search using Confab via Open Babel
https://open-babel.readthedocs.io/en/latest/3DStructureGen/multipleconformers.html. """
import os

from conformation.create_logger import create_logger
from conformation.confab import confab, Args


if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    confab(args, logger)
