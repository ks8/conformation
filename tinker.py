import argparse
import os
import numpy as np
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
import scipy.spatial


def tinker_md(args):
    """
    Generate MD simulation for molecules from SMILES string input
    :param args: Argparse arguments
    :return: None
    """
    for _, _, files in os.walk(args.input):
        for f in files:
            molecule_name = f[:f.find(".")]
            sdf_name = os.path.join(args.out, molecule_name + "." + "sdf")
            xyz_name = os.path.join(args.out, molecule_name + "." + "xyz")
            key_name = os.path.join(args.out, molecule_name + "." + "key")
            with open(os.path.join(args.input, f)) as tmp:
                smiles = tmp.readlines()[0].split()[0]
            # os.system("obabel --gen3D -ismi " + os.path.join(args.input, f) + " -osdf " + "-O " + sdf_name)
            counter = 0

            for j in range(args.num_starts):
                # m = Chem.MolFromSmiles(smiles)
                # m2 = Chem.AddHs(m)
                # _ = AllChem.EmbedMolecule(m2)
                # Chem.rdForceFieldHelpers.MMFFOptimizeMolecule(m2)

                m = Chem.MolFromSmiles(args.smiles)
                m2 = Chem.AddHs(m)
                _ = AllChem.EmbedMultipleConfs(m2, numConfs=args.num_configs)
                res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=args.max_iter)

                print(Chem.rdmolfiles.MolToPDBBlock(m2), file=open(molecule_name + ".pdb", "w+"))
                os.system("obabel -ipdb " + molecule_name + ".pdb" + " -osdf -O " + sdf_name)

                exit()

                with open(sdf_name, "r") as tmp:
                    contents = tmp.readlines()
                    contents[0] = molecule_name + "\n"
                with open(sdf_name, "w") as tmp:
                    for i in range(len(contents)):
                        tmp.write(contents[i])
                os.system("rm " + molecule_name + "." + "pdb ")
                os.system("sdf2tinkerxyz < " + sdf_name)
                with open(key_name, "w") as tmp:
                    tmp.write("parameters    " + args.param_path + "\n")
                    tmp.write("integrator    " + args.integrator + "\n")
                    tmp.write("archive" + "\n")
                os.system("mv " + molecule_name + "." + "xyz " + args.out)
                os.system("rm " + molecule_name + "." + "key ")
                os.system("dynamic " + xyz_name + " -k " + key_name + " " + str(args.num_steps) + " " +
                          str(args.time_step) + " " + str(args.save_step) + " " + str(args.ensemble) + " " +
                          str(args.temp))

            m = Chem.MolFromSmiles(smiles)
            m2 = Chem.AddHs(m)
            _ = AllChem.EmbedMultipleConfs(m2, numConfs=1)
            c = m2.GetConformers()[0]
            with open(os.path.join(args.out, molecule_name + ".arc"), "r") as tmp:
                line = tmp.readline()
                print(line)
                pos = []
                while line:
                    if line.split()[1] == molecule_name:
                        if counter > 0:
                            pos = np.array(pos)
                            np.savetxt(os.path.join(args.out, "pos", "pos-" + str(counter - 1) + "-" + molecule_name +
                                                    ".txt"), pos)
                            num_atoms = pos.shape[0]
                            dist_mat = np.zeros([num_atoms, num_atoms])
                            for i in range(num_atoms):
                                for j in range(1, num_atoms):
                                    if j > i:
                                        dist_mat[i][j] = scipy.spatial.distance.euclidean(pos[i], pos[j])
                                        dist_mat[j][i] = dist_mat[i][j]
                            np.savetxt(os.path.join(args.out, "distmat", "distmat-" + str(counter - 1) + "-" +
                                                    molecule_name + ".txt"), dist_mat)

                            for i in range(len(pos)):
                                c.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))
                            if args.dihedral:
                                dihedral = Chem.rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0],
                                                                               args.dihedral_vals[1],
                                                                               args.dihedral_vals[2],
                                                                               args.dihedral_vals[3])
                            else:
                                dihedral = "nan"
                            res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=0)
                            with open(os.path.join(args.out, "properties", "energy-rms-dihedral" + str(counter - 1) +
                                                                           "-" + molecule_name + ".txt"), "w") as o:
                                o.write("energy: " + str(res[0][1]))
                                o.write('\n')
                                o.write("rms: " + "nan")
                                o.write('\n')
                                o.write("dihedral: " + str(dihedral))
                        pos = []
                        counter += 1
                    else:
                        pos.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    line = tmp.readline()


def main():
    """
    Parse arguments and run tinkerMD function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None,
                        help='Folder path containing relevant input files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder path containing output files')
    parser.add_argument('--param_path', type=str, dest='param_path',
                        default="/data/swansonk1/anaconda3/envs/my-rdkit-env/Tinker-FFE/tinker/params/mmff",
                        help='File path to Tinker parameters')
    parser.add_argument('--integrator', type=str, dest='integrator', default="verlet",
                        help='File path to Tinker parameters')
    parser.add_argument('--num_steps', type=int, dest='num_steps', default=100000, help='Number of MD steps')
    parser.add_argument('--time_step', type=float, dest='time_step', default=1.0, help='Time step in femtoseconds')
    parser.add_argument('--save_step', type=float, dest='save_step', default=0.1, help='Time btw saves in picoseconds')
    parser.add_argument('--ensemble', type=int, dest='ensemble', default=2, help='1=NVE, 2=NVT')
    parser.add_argument('--temp', type=int, dest='temp', default=298, help='Temperature in degrees Kelvin')
    parser.add_argument('--dihedral', action='store_true', default=False,
                        help='Use when computing dihedral angle values')
    parser.add_argument('--dihedral_vals', type=int, dest='dihedral_vals', nargs='+', default=[2, 0, 1, 5],
                        help='Atom IDs for dihedral')
    parser.add_argument('--num_starts', type=int, dest='num_starts', default=1,
                        help='Number of MD restarts from low energy conformations generated by RDKit')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    os.makedirs(os.path.join(args.out, "pos"), exist_ok=False)
    os.makedirs(os.path.join(args.out, "properties"), exist_ok=False)
    os.makedirs(os.path.join(args.out, "distmat"), exist_ok=False)
    tinker_md(args)


if __name__ == '__main__':
    main()
