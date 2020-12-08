""" Run basic NF sampling. """
from conformation.run_basic_nf_sampling import run_basic_nf_sampling, Args

if __name__ == '__main__':
    run_basic_nf_sampling(Args().parse_args())
