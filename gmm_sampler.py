""" Samples from GMM """
from conformation.gmm_sampler import gmm_sampler, Args

if __name__ == '__main__':
    gmm_sampler(Args().parse_args())

