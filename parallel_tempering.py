""" General framework for parallel tempering MCMC. """
from conformation.parallel_tempering import parallel_tempering, Args

if __name__ == '__main__':
    parallel_tempering(Args().parse_args())
