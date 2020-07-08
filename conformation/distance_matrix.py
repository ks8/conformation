""" Compute pairwise distance matrix from a conformation. """
import numpy as np

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
    for i in range(num_atoms):
        for j in range(1, num_atoms):
            if j > i:
                dist_mat[i][j] = scipy.spatial.distance.euclidean(positions[i], positions[j])
                dist_mat[j][i] = dist_mat[i][j]
    np.savetxt(destination, dist_mat)


def distmat_to_vec(path: str) -> np.ndarray:
    """

    :param path:
    :return:
    """
    data = []
    distmat = np.loadtxt(path)
    num_atoms = distmat.shape[0]
    for m in range(num_atoms):
        for n in range(1, num_atoms):
            if n > m:
                data.append(distmat[m][n])
    return np.array(data)
