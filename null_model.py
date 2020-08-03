""" Null model for predicting pairwise distances based on minimum path length between pairs of atoms. """
from conformation.null_model import null_model, Args
from conformation.create_logger import create_logger

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    null_model(args, logger)
