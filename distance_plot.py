""" Plot distributions of atomic pairwise distances. """
from conformation.distance_plot import distance_plot, Args

if __name__ == '__main__':
    distance_plot(Args().parse_args())
