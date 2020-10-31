""" Run Schrodinger's MacroModel conformational search tools (LMOD and MCMM). """
import os

from conformation.create_logger import create_logger
from conformation.macromodel import macromodel, Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    args.save(os.path.join(args.save_dir, "args.json"))
    macromodel(args, logger)
