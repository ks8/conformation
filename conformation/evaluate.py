""" Compute evaluation metrics. """
import argparse
import itertools
import math
import numpy as np
import os
# noinspection PyUnresolvedReferences
from scipy.stats import entropy

# noinspection PyUnresolvedReferences
import ot
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem
import scipy.spatial


def evaluate(data_1: np.ndarray, data_2: np.ndarray, m1: Chem.rdchem.Mol, m2: Chem.rdchem.Mol,
             max_ot_samples: int, out: str, num_dihedral_bins: int) -> None:
    """
    Compute all pairwise correlation coefficients across columns of a numpy array.
    :param num_dihedral_bins:
    :param m2:
    :param m1:
    :param max_ot_samples:
    :param data_1:
    :param data_2:
    :param out:
    """

    corr_coef_1 = []
    for m, n in itertools.combinations(list(np.arange(data_1.shape[1])), 2):
        corr_coef_1.append(np.corrcoef(data_1[:, m], data_1[:, n])[0][1])
    corr_coef_1 = np.array(corr_coef_1)

    corr_coef_2 = []
    for m, n in itertools.combinations(list(np.arange(data_2.shape[1])), 2):
        corr_coef_2.append(np.corrcoef(data_2[:, m], data_2[:, n])[0][1])
    corr_coef_2 = np.array(corr_coef_2)

    max_samples = min(len(data_1), len(data_2), max_ot_samples)
    m = ot.dist(data_1[np.random.randint(0, len(data_1), max_samples), :], data_2[np.random.randint(0, len(data_2),
                                                                                                    max_samples), :])
    a = np.ones(max_samples)/float(max_samples)
    b = a
    g0 = ot.emd(a, b, m)
    emd = (g0*m).sum()

    dihedral_indices = []
    for w, x, y, z in itertools.combinations(list(np.arange(m1.GetNumAtoms())), 4):
        dihedral_indices.append([w, x, y, z])

    dihedral_vals = []
    for c in m1.GetConformers():
        for i in range(len(dihedral_indices)):
            dihedral_vals.append(Chem.rdMolTransforms.GetDihedralRad(c, dihedral_indices[i][0].item(),
                                                                     dihedral_indices[i][1].item(),
                                                                     dihedral_indices[i][2].item(),
                                                                     dihedral_indices[i][3].item()))
    dihedral_vals = np.array(dihedral_vals)
    dihedral_dist_1 = np.histogram(dihedral_vals, bins=[-math.pi + i*(2.*math.pi/float(num_dihedral_bins))
                                                        for i in range(num_dihedral_bins + 1)], density=True)[0]

    dihedral_vals = []
    for c in m2.GetConformers():
        for i in range(len(dihedral_indices)):
            dihedral_vals.append(Chem.rdMolTransforms.GetDihedralRad(c, dihedral_indices[i][0].item(),
                                                                     dihedral_indices[i][1].item(),
                                                                     dihedral_indices[i][2].item(),
                                                                     dihedral_indices[i][3].item()))
    dihedral_vals = np.array(dihedral_vals)
    dihedral_dist_2 = np.histogram(dihedral_vals, bins=[-math.pi + i*(2.*math.pi/float(num_dihedral_bins))
                                                        for i in range(num_dihedral_bins + 1)], density=True)[0]

    dihedral_dist_1 = np.where(dihedral_dist_1 == 0, 1e-10, dihedral_dist_1)
    dihedral_dist_2 = np.where(dihedral_dist_2 == 0, 1e-10, dihedral_dist_2)

    with open(out + ".txt", "w") as o:
        o.write("corr_coef: ")
        o.write(str(scipy.spatial.distance.euclidean(corr_coef_1, corr_coef_2)))
        o.write("\n")
        o.write("KL: ")
        o.write(str(entropy(dihedral_dist_1, dihedral_dist_2)))
        o.write("\n")
        o.write("OT cost: ")
        o.write(str(emd))
        o.write("\n")


def main():
    """
    Parse arguments and run run_training function.
    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--distmat_dir_1', type=str, dest='distmat_dir_1', default=None,
                        help='Path to one directory containing data')
    parser.add_argument('--distmat_dir_2', type=str, dest='distmat_dir_2', default=None,
                        help='Path to another directory containing data')
    parser.add_argument('--conf_path_1', type=str, dest='conf_path_1', default=None,
                        help='Path to another directory containing data')
    parser.add_argument('--conf_path_2', type=str, dest='conf_path_2', default=None,
                        help='Path to another directory containing data')
    parser.add_argument('--max_ot_samples', type=int, dest='max_ot_samples', default=2000,
                        help='Max # samples used for optimal transport cost')
    parser.add_argument('--num_dihedral_bins', type=int, dest='num_dihedral_bins', default=1000,
                        help='# histogram bins for dihedral angle distribution')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Path to output text file')
    args = parser.parse_args()

    distance_vectors_1 = []
    for _, _, files in os.walk(args.distmat_dir_1):
        for f in files:
            data = []
            distmat = np.loadtxt(os.path.join(args.distmat_dir_1, f))
            num_atoms = distmat.shape[0]
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        data.append(distmat[m][n])
            distance_vectors_1.append(data)
    distance_vectors_1 = np.array(distance_vectors_1)

    distance_vectors_2 = []
    for _, _, files in os.walk(args.distmat_dir_2):
        for f in files:
            data = []
            distmat = np.loadtxt(os.path.join(args.distmat_dir_2, f))
            num_atoms = distmat.shape[0]
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        data.append(distmat[m][n])
            distance_vectors_2.append(data)
    distance_vectors_2 = np.array(distance_vectors_2)

    m1 = AllChem.MolFromPDBFile(args.conf_path_1, removeHs=False)
    m2 = AllChem.MolFromPDBFile(args.conf_path_2, removeHs=False)

    evaluate(distance_vectors_1, distance_vectors_2, m1, m2, args.max_ot_samples, args.out, args.num_dihedral_bins)


if __name__ == '__main__':
    main()
