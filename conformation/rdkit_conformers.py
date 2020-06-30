""" Generate conformations using RDKit. """
import argparse
from argparse import Namespace
import numpy as np
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

    with open(os.path.join(args.out, "pos", "atoms.txt"), "w") as f:
        for atom in m2.GetAtoms():
            f.write(atom.GetSymbol() + '\n')

    i = 0
    for c in m2.GetConformers():
        pos = c.GetPositions()
        np.savetxt(os.path.join(args.out, "pos", "pos-" + str(i) + ".txt"), pos)
        dist_matrix(pos, os.path.join(args.out, "distmat", "distmat-" + str(i) + ".txt"))
        with open(os.path.join(args.out, "properties", "energy-rms-" + str(i) + ".txt"), "w") as f:
            f.write("energy: " + str(res[i][1]))
            f.write('\n')
            f.write("rms: " + str(rms_list[i]))
            f.write('\n')
            dihedral = Chem.rdMolTransforms.GetDihedralRad(c, args.dihedral[0], args.dihedral[1], args.dihedral[2],
                                                           args.dihedral[3])
            f.write("dihedral: " + str(dihedral))
        i += 1


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
    parser.add_argument('--dihedral', type=int, dest='dihedral', nargs='+', default=[2, 0, 1, 5],
                        help='Atom IDs for dihedral')

    args = parser.parse_args()

    os.makedirs(args.out)
    os.makedirs(os.path.join(args.out, "pos"))
    os.makedirs(os.path.join(args.out, "properties"))
    os.makedirs(os.path.join(args.out, "distmat"))
    conformers(args)


if __name__ == '__main__':
    main()
