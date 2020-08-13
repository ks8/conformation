""" QM9 to SMILES. Dataset from: http://moleculenet.ai/datasets-1,
code partially from: https://github.com/nyu-dl/dl4chem-geometry/blob/master/CSD_sdf_to_p.py"""
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdmolops
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to QM9 sdf file
    save_dir: str  # Path to directory for output files
    n_min: int = 2  # Minimum number of heavy atoms
    n_max: int = 9  # Maximum number of heavy atoms
    max_num: int = 10000  # Maximum number of molecules to read
    exclude_f: bool = True  # Whether or not to exclude F atoms


def qm9_to_smiles(args: Args) -> None:
    """

    :param args:
    """
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "smiles"))
    os.makedirs(os.path.join(args.save_dir, "binaries"))

    suppl = Chem.SDMolSupplier(args.data_path)
    counter = 0
    for i, mol in enumerate(suppl):
        if counter < args.max_num:
            # noinspection PyBroadException
            try:
                rdmolops.AssignAtomChiralTagsFromStructure(mol)
                rdmolops.AssignStereochemistry(mol)

                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                if '.' in smiles:
                    continue

            except:
                continue

            na = mol.GetNumHeavyAtoms()
            pos = mol.GetConformer().GetPositions()
            if na == pos.shape[0] and args.n_min <= na <= args.n_max:

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
                    counter += 1

                    bin_str = mol.ToBinary()
                    with open(os.path.join(args.save_dir, "binaries", "qm9_" + str(counter) + ".bin"), "wb") as f:
                        f.write(bin_str)

        else:
            break
