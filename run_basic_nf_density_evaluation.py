""" Run basic NF density evaluation. """
from conformation.run_basic_nf_density_evaluation import run_basic_nf_density_evaluation, Args

if __name__ == '__main__':
    run_basic_nf_density_evaluation(Args().parse_args())
