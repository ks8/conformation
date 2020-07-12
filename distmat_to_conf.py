""" Generate molecular conformations from atomic pairwise distance matrices. """
from conformation.distmat_to_conf import distmat_to_conf, Args

if __name__ == '__main__':
    distmat_to_conf(Args().parse_args())
