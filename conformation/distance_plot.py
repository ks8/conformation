""" Plot distributions of atomic pairwise distances. """
import argparse
from argparse import Namespace
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from conformation.distance_matrix import distmat_to_vec


def distance_plot(args: Namespace):
    """
    Plot histograms of pairwise distances for a set of molecular conformations.
    :param args: Argparse arguments.
    :return: None.
    """
    num_atoms = None
    distances = []
    for _, _, files in os.walk(args.input):
        for f in files:
            num_atoms, data = distmat_to_vec(os.path.join(args.input, f))
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
        plt.savefig(args.out + "-" + str(labels[i][0]) + "-" + str(labels[i][1]) + "-distances.png")
        plt.clf()


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None, help='Folder path containing relevant files')
    parser.add_argument('--num_bins', type=int, dest='num_bins', default=50,
                        help='# histogram bins')
    parser.add_argument('--out', type=str, dest='out', default=None, help='File name for plot image')
    args = parser.parse_args()

    distance_plot(args)


if __name__ == '__main__':
    main()
