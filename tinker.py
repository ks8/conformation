""" Run Tinker MD simulations and extract conformations. """
from conformation.tinker import tinker, Args

if __name__ == '__main__':
    args = Args().parse_args()
    tinker(args)
