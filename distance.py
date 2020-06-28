import scipy.spatial
import numpy as np


def dist_matrix(positions, destination):
    """
    Compute the pairwise distance matrix for a molecule and save as a text file in a specified file
    :param positions: numpy array of atomic coordinates
    :param destination: output file
    :return: None
    """
    num_atoms = positions.shape[0]
    dist_mat = np.zeros([num_atoms, num_atoms])
    for i in range(num_atoms):
        for j in range(1, num_atoms):
            if j > i:
                dist_mat[i][j] = scipy.spatial.distance.euclidean(positions[i], positions[j])
                dist_mat[j][i] = dist_mat[i][j]
    np.savetxt(destination, dist_mat)
