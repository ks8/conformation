""" Run Graph Normalizing Flow network training. """
from conformation.create_logger import create_logger
from conformation.run_gnf_training import run_gnf_training
from conformation.train_args_relational import Args

if __name__ == '__main__':
    args = Args().parse_args()
    logger = create_logger(name='train', save_dir=args.save_dir)
    run_gnf_training(args, logger)
