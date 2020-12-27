""" Compute 2D KL divergence. """
from conformation.kl_divergence import kl_divergence, Args

if __name__ == '__main__':
    kl_divergence(Args().parse_args())
