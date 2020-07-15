""" Generate a molecular conformation from an atomic pairwise distance matrix. """
import itertools
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing all distance matrices
    save_dir: str  # Path to directory containing output files
    smiles: str  # SMILES string
    offset: float = 0.0005  # Offset for bounds matrix


def distmat_to_conf(args: Args) -> None:
    """
    Generate a molecular conformation from an atomic pairwise distance matrix.
    :param args: System arguments.
    :return: None.
    """

    os.makedirs(args.save_dir)

    # Conformation counter
    counter = 0

    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDG()

    # Create a random conformation object
    tmp = Chem.MolFromSmiles(args.smiles)
    tmp = Chem.AddHs(tmp)

    for _, _, files in os.walk(args.data_dir):
        for f in files:
            # Use the pairwise distance matrix to set the ETKDG bounds matrix
            dist_mat = np.loadtxt(os.path.join(args.data_dir, f))
            num_atoms = dist_mat.shape[0]
            for i, j in itertools.combinations(np.arange(num_atoms), 2):
                dist_mat[i][j] += args.offset
                dist_mat[j][i] -= args.offset
            ps.SetBoundsMat(dist_mat)

            AllChem.EmbedMolecule(tmp, params=ps)

            try:
                # Test that the conformation is valid
                c = tmp.GetConformer()

                # Set the conformer Id and increment the conformation counter
                c.SetId(counter)
                counter += 1

                # Add the conformer to the overall molecule object
                mol.AddConformer(c)

            except ValueError:
                continue

    # Print the conformations to a binary file
    bin_str = mol.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)
