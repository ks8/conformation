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
    n_max: int = 20  # Maximum number of heavy atoms  # TODO: I think QM9 has a global max of 9 heavy atoms...
    max_num: int = 20  # Maximum number of molecules to read


def qm9_to_smiles(args: Args) -> None:
    """

    :param args:
    """
    os.makedirs(args.save_dir)

    suppl = Chem.SDMolSupplier(args.data_path)
    counter = 0
    for i, mol in enumerate(suppl):
        if counter < args.max_num:
            # noinspection PyBroadException
            try:
                rdmolops.AssignAtomChiralTagsFromStructure(mol)
                rdmolops.AssignStereochemistry(mol)
                AllChem.EmbedMolecule(mol)  # TODO: Why does this sometimes fail?
                mol.GetConformer()
                smiles = Chem.MolToSmiles(mol)
                na = mol.GetNumHeavyAtoms()
                pos = mol.GetConformer().GetPositions()
                if na == pos.shape[0] and args.n_min <= na <= args.n_max:
                    with open(os.path.join(args.save_dir, "qm9_" + str(counter) + ".smiles"), "w") as f:
                        f.write(smiles)
                    counter += 1
            except:
                continue
        else:
            break
