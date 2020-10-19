""" Plotting of conformation distributions. """
from conformation.analyze_distributions import analyze_distributions, Args

if __name__ == '__main__':
    analyze_distributions(Args().parse_args())
