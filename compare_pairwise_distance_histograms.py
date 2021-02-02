""" Plot matrix of pairwise distance histograms for two sets of conformations. """
from conformation.compare_pairwise_distance_histograms import compare_pairwise_distance_histograms, Args

if __name__ == '__main__':
    compare_pairwise_distance_histograms(Args().parse_args())
