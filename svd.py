""" Compute distribution of singular values for distance matrices. """
from conformation.svd import svd, Args

if __name__ == '__main__':
    svd(Args().parse_args())
