""" Compute mean and variance of atomic pairwise distances across MD-generated conformations of a set of molecules. """
from conformation.average_conformation_distances import average_conformation_distances, Args

if __name__ == '__main__':
    average_conformation_distances(Args().parse_args())
