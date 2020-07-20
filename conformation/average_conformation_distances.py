""" Compute mean and standard deviation of atomic pairwise distances across MD-generated conformations of a set of
molecules. """
import numpy as np
import os
import re

# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import distmat_to_vec


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing all distance matrices
    save_dir: str  # Path to directory containing output files (average distances)


def average_conformation_distances(args: Args) -> None:
    """
    Compute mean and variance of atomic pairwise distances across MD-generated conformations of a set of molecules.
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)
    avg_std_dist = dict()
    for _, _, files_dist in os.walk(args.data_dir):
        for f_dist in files_dist:
            molecule_name = f_dist[[m.start() for m in re.finditer("-", f_dist)][1] + 1:f_dist.find(".")]
            _, dist_vec = distmat_to_vec(os.path.join(args.data_dir, f_dist))
            if molecule_name in avg_std_dist:
                avg_std_dist[molecule_name].append(dist_vec)
            else:
                avg_std_dist[molecule_name] = []
                avg_std_dist[molecule_name].append(dist_vec)

    for _, mol in enumerate(avg_std_dist):
        arr = np.array(avg_std_dist[mol])
        avg = arr.mean(axis=0)
        std = arr.std(axis=0)
        avg_std = np.concatenate((avg.reshape(avg.shape[0], 1), std.reshape(std.shape[0], 1)), axis=1)
        np.save(os.path.join(args.save_dir, "avg-std-" + mol), avg_std)
