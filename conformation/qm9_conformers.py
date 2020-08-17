""" Extract QM9 conformations and generate distance matrices. """
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdmolops
# noinspection PyPackageRequirements
from tap import Tap

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to QM9 sdf file
    save_dir: str  # Directory for saving conformations
    n_min: int = 2  # Minimum number of heavy atoms
    n_max: int = 9  # Maximum number of heavy atoms
    max_num: int = 10000  # Maximum number of molecules to read
    exclude_f: bool = False  # Whether or not to exclude F atoms


def qm9_conformers(args: Args) -> None:
    """
    Extract QM9 conformations.
    :param args: Argparse arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "smiles"))
    os.makedirs(os.path.join(args.save_dir, "binaries"))
    os.makedirs(os.path.join(args.save_dir, "distmat"))

    suppl = Chem.SDMolSupplier(args.data_path, removeHs=False)
    counter = 0
    for i, mol in enumerate(suppl):
        if counter < args.max_num:
            # noinspection PyBroadException
            try:
                rdmolops.AssignAtomChiralTagsFromStructure(mol)
                rdmolops.AssignStereochemistry(mol)

                mol = Chem.AddHs(mol, addCoords=True)

                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                if '.' in smiles:
                    continue

            except:
                continue

            na = mol.GetNumHeavyAtoms()
            if args.n_min <= na <= args.n_max:
                # Exclude molecule if it contains any F atoms
                f_present = False
                if args.exclude_f:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 9:
                            f_present = True
                            break

                if not f_present:
                    with open(os.path.join(args.save_dir, "smiles", "qm9_" + str(counter) + ".smiles"), "w") as f:
                        f.write(smiles)

                    bin_str = mol.ToBinary()
                    with open(os.path.join(args.save_dir, "binaries", "qm9_" + str(counter) + ".bin"), "wb") as f:
                        f.write(bin_str)

                    pos = mol.GetConformer().GetPositions()
                    dist_matrix(pos, os.path.join(args.save_dir, "distmat", "distmat-lowenergy-qm9_" + str(counter)))

                    counter += 1

        else:
            break
