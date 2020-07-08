""" QM9 to SMILES. Dataset from: http://moleculenet.ai/datasets-1, code partially from: https://github.com/nyu-dl/dl4chem-geometry/blob/master/CSD_sdf_to_p.py"""
import argparse
from argparse import Namespace
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem


def qm9_to_smiles(args: Namespace) -> None:
    """

    :param args:
    """
    suppl = Chem.SDMolSupplier(args.data_path)
    counter = 0
    for i, mol in enumerate(suppl):
        if counter < args.max_num:
            # noinspection PyBroadException
            try:
                Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
                Chem.rdmolops.AssignStereochemistry(mol)
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


def main():
    """
    Parse arguments and run run_training function.
    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_min', type=int, dest='n_min', default=2, help='Min # heavy atoms')
    parser.add_argument('--n_max', type=int, dest='n_max', default=20, help='Max # heavy atoms')
    parser.add_argument('--max_num', type=int, dest='max_num', default=20, help='Max # molecules to read')
    parser.add_argument('--data_path', type=str, dest='data_path', default=None, help='Path to QM9 sdf file')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default=None, help='Directory to save output files')
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    qm9_to_smiles(args)


if __name__ == '__main__':
    main()
