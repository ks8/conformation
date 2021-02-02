""" Generate distance matrices from molecular conformations. """
from conformation.check_bounds import conf_to_distmat, Args

if __name__ == '__main__':
    conf_to_distmat(Args().parse_args())
