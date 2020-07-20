""" Plot distributions of atomic pairwise distances. """
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import distmat_to_vec


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing all distance matrices
    save_path: str  # Output file name
    num_bins: int = 50  # Number of histogram bins


def distance_plot(args: Args):
    """
    Plot histograms of pairwise distances for a set of molecular conformations.
    :param args: Argparse arguments.
    :return: None.
    """
    num_atoms = None
    distances = []
    for _, _, files in os.walk(args.data_dir):
        for f in files:
            num_atoms, data = distmat_to_vec(os.path.join(args.data_dir, f))
            distances.append(data)

    labels = []
    for i, j in itertools.combinations(np.arange(num_atoms), 2):
        labels.append([i, j])

    distances = np.array(distances)
    for i in range(distances.shape[1]):
        plt.hist(distances[:, i], bins=args.num_bins)
        plt.title(str(labels[i][0]) + "-" + str(labels[i][1]) + " Distances")
        plt.ylabel("Frequency")
        plt.xlabel("Distance ($\AA$)")
        plt.savefig(args.save_path + "-" + str(labels[i][0]) + "-" + str(labels[i][1]) + "-distances.png")
        plt.clf()
