import numpy as np
import argparse
import os
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D


def tinker_to_pos(args):
    """
    Convert tinker MD trajectories to conformation files
    :param args: Argparse arguments
    :return: None
    """
    for _, _, files in os.walk(args.input):
        for f in files:
            counter = 0
            if f[-3:] == "arc":
                with open(os.path.join(args.folder, f[:-4] + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                m = Chem.MolFromSmiles(smiles)
                m2 = Chem.AddHs(m)
                _ = AllChem.EmbedMultipleConfs(m2, numConfs=1)
                c = m2.GetConformers()[0]
                with open(os.path.join(args.input, f), "r") as tmp:
                    line = tmp.readline()
                    pos = []
                    while line:
                        if line.split()[1] == f[:-4]:
                            if counter > 0:
                                pos = np.array(pos)
                                np.savetxt(os.path.join(args.out, "pos-" + str(counter - 1) + "-" + f[:-4] + ".txt"),
                                           pos)
                                with open(os.path.join(args.out, "energy-rms-" + str(counter - 1) + "-" + f[:-4] +
                                                                 ".txt"), "w") as o:
                                    for i in range(len(pos)):
                                        c.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))
                                    dihedral = Chem.rdMolTransforms.GetDihedralRad(c, args.dihedral[0],
                                                                                   args.dihedral[1], args.dihedral[2],
                                                                                   args.dihedral[3])
                                    o.write("dihedral: " + str(dihedral))
                            pos = []
                            counter += 1
                        else:
                            pos.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                        line = tmp.readline()


def main():
    """
    Parse arguments and run conformers function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--input', type=str, dest='input', default=None, help='Folder name containing input files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder name for saving output')
    parser.add_argument('--dihedral', type=int, dest='dihedral', nargs='+', default=[2, 0, 1, 5],
                        help='Atom IDs for dihedral')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    tinker_to_pos(args)


if __name__ == '__main__':
    main()