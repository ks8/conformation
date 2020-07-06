""" Compute pairwise correlations for atomic pairwise distances across a set of distance matrices. """
import argparse
import itertools
import numpy as np
import os
# noinspection PyUnresolvedReferences
import ot
import scipy.spatial


def evaluate(data_1: np.ndarray, data_2: np.ndarray, out: str) -> None:
    """
    Compute all pairwise correlation coefficients across columns of a numpy array.
    :param data_1:
    :param data_2:
    :param out:
    """

    corr_coef_1 = []
    for m, n in itertools.combinations(list(np.arange(data_1.shape[1])), 2):
        corr_coef_1.append(np.corrcoef(data_1[:, m], data_1[:, n])[0][1])
    corr_coef_1 = np.array(corr_coef_1)

    corr_coef_2 = []
    for m, n in itertools.combinations(list(np.arange(data_1.shape[1])), 2):
        corr_coef_2.append(np.corrcoef(data_1[:, m], data_1[:, n])[0][1])
    corr_coef_2 = np.array(corr_coef_2)

    max_samples = min(len(data_1), len(data_2))
    m = ot.dist(data_1[np.random.randint(0, len(data_1), max_samples), :], data_2[np.random.randint(0, len(data_2),
                                                                                                    max_samples), :])
    a = np.ones(max_samples)/float(max_samples)
    b = a
    g0 = ot.emd(a, b, m)
    emd = (g0*m).sum()
    
    with open(out, "w") as o:
        o.write("corr_coef: ")
        o.write(str(scipy.spatial.distance.euclidean(corr_coef_1, corr_coef_2)))
        o.write("\n")
        o.write("OT cost: ")
        ot.write(str(emd))
        ot.write("\n")


def main():
    """
    Parse arguments and run run_training function.
    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_1', type=str, dest='data_dir_1', default=None,
                        help='Path to one directory containing data')
    parser.add_argument('--data_dir_2', type=str, dest='data_dir_2', default=None,
                        help='Path to another directory containing data')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Path to output text file')
    args = parser.parse_args()

    distance_vectors_1 = []
    for _, _, files in os.walk(args.data_dir_1):
        for f in files:
            data = []
            distmat = np.loadtxt(os.path.join(args.data_dir_1, f))
            num_atoms = distmat.shape[0]
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        data.append(distmat[m][n])
            distance_vectors_1.append(data)
    distance_vectors = np.array(distance_vectors_1)

    distance_vectors_2 = []
    for _, _, files in os.walk(args.data_dir_1):
        for f in files:
            data = []
            distmat = np.loadtxt(os.path.join(args.data_dir_2, f))
            num_atoms = distmat.shape[0]
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        data.append(distmat[m][n])
            distance_vectors_2.append(data)
    distance_vectors_2 = np.array(distance_vectors_2)

    evaluate(distance_vectors_1, distance_vectors_2, args.out)


if __name__ == '__main__':
    main()
