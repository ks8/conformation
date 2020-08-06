""" Embed molecules from smiles strings. """
import os
import re

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing SMILES files
    save_dir: str  # Directory for saving embedded molecules
    max_attempts: int = 100000  # Maximum attempts for 3D embedding


def embed_molecules(args: Args) -> None:
    """
    3D embedding of molecules
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)

    for root, _, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(root, f)
            molecule_name = f[:f.find(".")]
            with open(path) as tmp:
                smiles = tmp.readlines()[0].split()[0]
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)

            try:
                AllChem.EmbedMolecule(mol, maxAttempts=args.max_attempts)
                AllChem.MMFFOptimizeMolecule(mol)

                bin_str = mol.ToBinary()
                with open(os.path.join(args.save_dir, molecule_name + ".bin"), "wb") as tmp:
                    tmp.write(bin_str)
            except ValueError:
                continue
