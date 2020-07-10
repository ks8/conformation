""" Generate a molecular conformation from an atomic pairwise distance matrix. """
import argparse
import itertools
from argparse import Namespace
import numpy as np
import os

# noinspection PyUnresolvedReferences
import rdkit
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdDistGeom


def distmat_to_conf(smiles: str, path: str, out: str, offset: float = 0.0005) -> None:
    """
    Generate conformation from distance matrix.
    :param out: Output file path
    :param offset: Add offset to upper triangle and subtract from lower to generate tight distance bounds matrix.
    :param path: Path to distance matrix text file.
    :param smiles: Molecular SMILES string.
    :return: None.
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDG()

    # Use the pairwise distance matrix to set the ETKDG bounds matrix
    distmat = np.loadtxt(path)
    num_atoms = distmat.shape[0]
    for i, j in itertools.combinations(np.arange(num_atoms), 2):
        distmat[i][j] += offset
        distmat[j][i] -= offset
    ps.SetBoundsMat(distmat)

    # Generate and print conformation as PDB file
    AllChem.EmbedMolecule(mol, params=ps)
    print(Chem.rdmolfiles.MolToPDBBlock(mol), file=open(out, "w+"))


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None, help='Directory containing input files')
    parser.add_argument('--smiles', type=str, dest='smiles', default=None, help='SMILES string')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Directory for output files')
    args = parser.parse_args()

    os.makedirs(args.out)

    for _, _, files in os.walk(args.input):
        for f in files:
            path = os.path.join(args.input, f)
            distmat_to_conf(args.smiles, path, os.path.join(args.out, "conf-" + f))


if __name__ == '__main__':
    main()
