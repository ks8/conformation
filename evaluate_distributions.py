""" Compute evaluation metrics. """
from conformation.evaluate_distributions import evaluate_distributions, Args

if __name__ == '__main__':
    evaluate_distributions(Args().parse_args())
