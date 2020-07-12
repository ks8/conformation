""" Compute pairwise distance matrix from a conformation. """
import itertools
import numpy as np
from typing import Tuple

import scipy.spatial


def dist_matrix(positions: np.ndarray, destination: str) -> None:
    """
    Compute the pairwise distance matrix for a molecule and save as a text file in a specified file.
    :param positions: numpy array of atomic coordinates.
    :param destination: output file.
    :return: None.
    """
    num_atoms = positions.shape[0]
    dist_mat = np.zeros([num_atoms, num_atoms])
    for i, j in itertools.combinations(np.arange(num_atoms), 2):
        dist_mat[i][j] = scipy.spatial.distance.euclidean(positions[i], positions[j])
        dist_mat[j][i] = dist_mat[i][j]
    # noinspection PyTypeChecker
    np.savetxt(destination, dist_mat)


def distmat_to_vec(path: str) -> Tuple[int, np.ndarray]:
    """
    Extract the upper triangle of a distance matrix and return it as a 1-D numpy array.
    :param path: Path to file containing numpy distance matrix.
    :return: 1-D numpy array containing the upper triangle of the distance matrix.
    """
    dist_mat = np.loadtxt(path)
    num_atoms = dist_mat.shape[0]
    dist_vec = dist_mat[np.triu_indices(dist_mat.shape[1], k=1)]

    return num_atoms, dist_vec
