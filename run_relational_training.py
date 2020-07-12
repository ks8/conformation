""" Run relational network training. """
from conformation.run_relational_training import run_relational_training, Args

if __name__ == '__main__':
    run_relational_training(Args().parse_args())
