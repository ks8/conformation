""" Generate conformations using RDKit. """
import argparse
from argparse import Namespace
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem

from conformation.distance_matrix import dist_matrix


def conformers(args: Namespace) -> None:
    """
    Generate conformations for a molecule.
    :param args: Argparse arguments.
    :return: None.
    """
    m = Chem.MolFromSmiles(args.smiles)
    m2 = Chem.AddHs(m)
    _ = AllChem.EmbedMultipleConfs(m2, numConfs=args.num_configs)
    res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=args.max_iter)
    rms_list = [0.0]
    AllChem.AlignMolConformers(m2, RMSlist=rms_list)

    i = 0
    for c in m2.GetConformers():
        pos = c.GetPositions()
        dist_matrix(pos, os.path.join(args.out, "distmat", "distmat-" + str(i) + ".txt"))
        with open(os.path.join(args.out, "properties", "energy-rms-dihedral-" + str(i) + ".txt"), "w") as f:
            f.write("energy: " + str(res[i][1]))
            f.write('\n')
            f.write("rms: " + str(rms_list[i]))
            f.write('\n')
            if args.dihedral:
                dihedral = Chem.rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0], args.dihedral_vals[1],
                                                               args.dihedral_vals[2], args.dihedral_vals[3])
            else:
                dihedral = "nan"
            f.write("dihedral: " + str(dihedral))
        i += 1

    # Print the conformations to a PDB file
    print(Chem.rdmolfiles.MolToPDBBlock(m2), file=open(os.path.join(args.out, "conformations.pdb"), "w+"))


def main():
    """
    Parse arguments and run conformers function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, dest='smiles', default=None, help='Molecule SMILES string')
    parser.add_argument('--num_configs', type=int, dest='num_configs', default=None, help='Number of conformations')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder name for saving conformations')
    parser.add_argument('--max_iter', type=int, dest='max_iter', default=200, help='Max iter for MMFF optimization')
    parser.add_argument('--dihedral', action='store_true', default=False,
                        help='Use when computing dihedral angle values')
    parser.add_argument('--dihedral_vals', type=int, dest='dihedral_vals', nargs='+', default=[2, 0, 1, 5],
                        help='Atom IDs for dihedral')

    args = parser.parse_args()

    os.makedirs(args.out)
    os.makedirs(os.path.join(args.out, "properties"))
    os.makedirs(os.path.join(args.out, "distmat"))
    conformers(args)


if __name__ == '__main__':
    main()
