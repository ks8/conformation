""" Compute evaluation metrics. """
import itertools
import math
import numpy as np
import os
# noinspection PyUnresolvedReferences
from scipy.stats import entropy
from typing import List

# noinspection PyUnresolvedReferences
import ot
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem
import scipy.spatial
# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import distmat_to_vec


def load_dist_matrices(data_dir: str) -> np.ndarray:
    """

    :param data_dir:
    :return:
    """
    distance_vectors = []
    for _, _, files in os.walk(data_dir):
        for f in files:
            _, data = distmat_to_vec(os.path.join(data_dir, f))
            distance_vectors.append(data)
    return np.array(distance_vectors)


def dihedral_histogram(dihedral_indices: List, conformations: np.ndarray, num_dihedral_bins: int) -> np.ndarray:
    """

    :param dihedral_indices:
    :param conformations:
    :param num_dihedral_bins:
    """
    dihedral_vals = []
    for c in conformations:
        for i in range(len(dihedral_indices)):
            dihedral_vals.append(Chem.rdMolTransforms.GetDihedralRad(c, dihedral_indices[i][0].item(),
                                                                     dihedral_indices[i][1].item(),
                                                                     dihedral_indices[i][2].item(),
                                                                     dihedral_indices[i][3].item()))
    dihedral_vals = np.array(dihedral_vals)
    histogram = np.histogram(dihedral_vals, bins=[-math.pi + i*(2.*math.pi/float(num_dihedral_bins))
                                                  for i in range(num_dihedral_bins + 1)], density=True)[0]
    return np.where(histogram == 0, 1e-10, histogram)


def evaluate(data_1: np.ndarray, data_2: np.ndarray, m1: np.ndarray, m2: np.ndarray,
             out: str, num_dihedral_bins: int, num_atoms: int) -> None:
    """
    Compute all pairwise correlation coefficients across columns of a numpy array.
    :param num_atoms:
    :param num_dihedral_bins:
    :param m2:
    :param m1:
    :param data_1:
    :param data_2:
    :param out:
    """

    # Compute Euclidean distance between pairwise correlation coefficient vectors
    corr_coef_1 = []
    corr_coef_2 = []
    for m, n in itertools.combinations(list(np.arange(data_1.shape[1])), 2):
        corr_coef_1.append(np.corrcoef(data_1[:, m], data_1[:, n])[0][1])
        corr_coef_2.append(np.corrcoef(data_2[:, m], data_2[:, n])[0][1])
    corr_coef_1 = np.array(corr_coef_1)
    corr_coef_2 = np.array(corr_coef_2)

    # Compute Optimal Transport cost (EMD)
    m = ot.dist(data_1, data_2)
    a = np.ones(len(data_1))/float(len(data_1))
    b = a
    g0 = ot.emd(a, b, m)
    emd = (g0*m).sum()

    # Compute KL divergence between histograms of dihedral angles computed from all quadruples of atoms
    # First, compute all 4-combinations
    dihedral_indices = []
    for w, x, y, z in itertools.combinations(list(np.arange(num_atoms)), 4):
        dihedral_indices.append([w, x, y, z])

    # Then, compute the histograms
    dihedral_dist_1 = dihedral_histogram(dihedral_indices, m1, num_dihedral_bins)
    dihedral_dist_2 = dihedral_histogram(dihedral_indices, m2, num_dihedral_bins)

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


class Args(Tap):
    """
    test
    """
    distmat_dir_1: str  # Path to one directory containing distance matrices
    distmat_dir_2: str  # Path to second directory containing distance matrices
    conf_path_1: str  # Path to one PDB file containing conformations
    conf_path_2: str  # Path to second PDB file containing conformations
    max_samples: int = 2000  # Max # samples used for evaluation
    num_dihedral_bins: int = 1000  # Number of histogram bins used for dihedral angle distribution
    out: str  # Path to output text file


def main(args: Args):
    """
    Parse arguments and run run_training function.
    :return: None.
    """

    # Load distance matrices
    distance_vectors_1 = load_dist_matrices(args.distmat_dir_1)
    distance_vectors_2 = load_dist_matrices(args.distmat_dir_2)

    # Load conformations
    m1 = AllChem.MolFromPDBFile(args.conf_path_1, removeHs=False)
    m2 = AllChem.MolFromPDBFile(args.conf_path_2, removeHs=False)
    conformations_1 = np.array(list(m1.GetConformers()))
    conformations_2 = np.array(list(m2.GetConformers()))

    # Get num atoms
    num_atoms = m1.GetNumAtoms()

    # Random sampling
    samples_1 = np.random.randint(0, len(distance_vectors_1), args.max_samples)
    samples_2 = np.random.randint(0, len(distance_vectors_2), args.max_samples)

    distance_vectors_1 = distance_vectors_1[samples_1, :]
    distance_vectors_2 = distance_vectors_2[samples_2, :]

    conformations_1 = conformations_1[samples_1]
    conformations_2 = conformations_2[samples_2]

    # Run evaluation
    evaluate(distance_vectors_1, distance_vectors_2, conformations_1, conformations_2, args.out, args.num_dihedral_bins,
             num_atoms)
