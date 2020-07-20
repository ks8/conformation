""" Run neural network sampling. """
from conformation.run_sampling import run_sampling, Args

if __name__ == '__main__':
    args = Args().parse_args()
    run_sampling(args)
