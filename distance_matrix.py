import argparse
import scipy.spatial
import os
import numpy as np


def dist(args):
    """
    Compute the pairwise distance matrix for a molecule and save as a text file in specified out folder
    :param args: Argparse arguments
    :return: None
    """
    for _, _, files in os.walk(args.folder):
        for f in files:
            if f[:3] == "pos":
                pos = np.loadtxt(os.path.join(args.folder, f))
                num_atoms = pos.shape[0]
                dist_mat = np.zeros([num_atoms, num_atoms])
                for i in range(num_atoms):
                    for j in range(1, num_atoms):
                        if j > i:
                            dist_mat[i][j] = scipy.spatial.distance.euclidean(pos[i], pos[j])
                            dist_mat[j][i] = dist_mat[i][j]
                np.savetxt(os.path.join(args.out, f), dist_mat)


def main():
    """
    Parse arguments and execute file processing
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder name to hold distance matrix files')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    dist(args)


if __name__ == '__main__':
    main()
