""" Compute mean and variance of atomic pairwise distances across MD-generated conformations of a set of
QM9 molecules. """
import numpy as np
import os

# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import distmat_to_vec


class Args(Tap):
    """
    System arguments.
    """
    distmat_dir: str  # Path to directory containing all distance matrices
    save_dir: str  # Path to directory containing output files (average distances)


def average_conformation_distances(args: Args) -> None:
    """
    Compute mean and variance of atomic pairwise distances across MD-generated conformations of a set of molecules.
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)
    avg_std_dist = dict()
    for _, _, files_dist in os.walk(args.distmat_dir):
        for f_dist in files_dist:
            molecule_name = f_dist[f_dist.find("qm9"):f_dist.find(".")]
            _, dist_vec = distmat_to_vec(os.path.join(args.distmat_dir, f_dist))
            if molecule_name in avg_std_dist:
                avg_std_dist[molecule_name].append(dist_vec)
            else:
                avg_std_dist[molecule_name] = []
                avg_std_dist[molecule_name].append(dist_vec)

    for _, mol in enumerate(avg_std_dist):
        avg = np.array(avg_std_dist[mol]).mean(axis=0)
        np.savetxt(os.path.join(args.save_dir, mol) + ".txt", avg)
