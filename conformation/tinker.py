""" Run Tinker MD simulations and extract conformations. """
import argparse
from argparse import Namespace
import numpy as np
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D

from conformation.distance_matrix import dist_matrix


def tinker_md(args: Namespace) -> None:
    """
    Generate MD simulation and conformations for molecules from SMILES string input.
    :param args: Argparse arguments.
    :return: None.
    """
    for _, _, files in os.walk(args.input):
        for f in files:
            # Set molecule, file, and smiles variables
            molecule_name = f[:f.find(".")]
            sdf_name = os.path.join(args.out, molecule_name + "." + "sdf")
            key_name = os.path.join(args.out, molecule_name + "." + "key")
            with open(os.path.join(args.input, f)) as tmp:
                smiles = tmp.readlines()[0].split()[0]

            # Initialize conformation counter
            counter = 0

            # Generate initial MMFF-minimized RDKit conformations from which to start MD simulations
            m = Chem.MolFromSmiles(smiles)
            m2 = Chem.AddHs(m)
            _ = AllChem.EmbedMultipleConfs(m2, numConfs=args.num_starts)
            _ = AllChem.MMFFOptimizeMoleculeConfs(m2,)
            print(Chem.rdmolfiles.MolToPDBBlock(m2), file=open(molecule_name, "w+"))

            # Convert PDB file to SDF file and remove PDB file
            os.system("obabel -ipdb " + molecule_name + " -osdf -O " + sdf_name)
            os.system("rm " + molecule_name)

            # Convert SDF file to multiple Tinker xyz input files
            os.system("sdf2tinkerxyz < " + sdf_name)

            # Label the first xyz file to match the other numbered labels (i.e., ethane.xyz -> ethane_1.xyz)
            os.system("mv " + molecule_name + "." + "xyz " + molecule_name + "_" +
                      ("{:0" + str(len(str(args.num_starts))) + "d}").format(1) + "." + "xyz ")
            os.system("mv " + molecule_name + "." + "key " + molecule_name + "_" +
                      ("{:0" + str(len(str(args.num_starts))) + "d}").format(1) + "." + "key ")

            # Write the key file, specifying MD simulation parameters
            with open(key_name, "w") as tmp:
                tmp.write("parameters    " + args.param_path + "\n")
                tmp.write("integrator    " + args.integrator + "\n")
                tmp.write("archive" + "\n")
                if args.thread_set:
                    tmp.write("openmp-threads    " + str(args.num_threads) + "\n")

            # Move the xyz files to the args.out folder
            os.system("mv " + molecule_name + "*." + "xyz " + args.out)

            # Remove extraneous key files
            os.system("rm " + molecule_name + "*." + "key ")

            # Run MD simulations and conformation extraction for each RDKit initial configuration
            for j in range(args.num_starts):
                # Run the MD simulation
                os.system("dynamic " + os.path.join(args.out, molecule_name + "_" + ("{:0" + str(len(str(args.num_starts
                                                                                                         ))) + "d}").
                                                    format(j + 1) + ".xyz") + " -k " + key_name + " " +
                          str(args.num_steps) + " " + str(args.time_step) + " " + str(args.save_step) + " " +
                          str(args.ensemble) + " " + str(args.temp))

                # Create a random conformation object, used for computing properties such as energy, dihedral angle
                m = Chem.MolFromSmiles(smiles)
                m2 = Chem.AddHs(m)
                _ = AllChem.EmbedMultipleConfs(m2, numConfs=1)
                c = m2.GetConformers()[0]

                # Open the trajectory (.arc) file and process the conformations
                with open(os.path.join(args.out, molecule_name + "_" + ("{:0" + str(len(str(args.num_starts))) +
                                                                        "d}").format(j + 1) + ".arc")) as tmp:
                    line = tmp.readline()
                    pos = []
                    new_file = True
                    while line:
                        # If we have finished extracting coordinates for one conformation, process the conformation
                        if line.split()[1] == molecule_name:
                            if not new_file:
                                # Save the atomic positions to a text file in the "pos" folder
                                pos = np.array(pos)
                                np.savetxt(os.path.join(args.out, "pos", "pos-" + str(counter) + "-" +
                                                        molecule_name + ".txt"), pos)

                                # Compute pairwise distance matrix and save to a text file in the "distmat" folder
                                dist_matrix(pos, os.path.join(args.out, "distmat", "distmat-" + str(counter) + "-"
                                                              + molecule_name + ".txt"))

                                # Save atomic coordinates to the conformation object
                                for i in range(len(pos)):
                                    c.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))

                                # Compute the specified dihedral angle
                                if args.dihedral:
                                    dihedral = Chem.rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0],
                                                                                   args.dihedral_vals[1],
                                                                                   args.dihedral_vals[2],
                                                                                   args.dihedral_vals[3])
                                else:
                                    dihedral = "nan"

                                # Compute the potential energy of the conformation
                                res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=0)

                                # Write property information to a text file in the "properties" folder
                                with open(os.path.join(args.out, "properties", "energy-rms-dihedral-" +
                                                                               str(counter) + "-" + molecule_name +
                                                                               ".txt"), "w") as o:
                                    o.write("energy: " + str(res[0][1]))
                                    o.write('\n')
                                    o.write("rms: " + "nan")
                                    o.write('\n')
                                    o.write("dihedral: " + str(dihedral))

                                pos = []
                                counter += 1

                        # Continue extracting coordinates for a single conformation
                        else:
                            new_file = False
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
    parser.add_argument('--thread_set', action='store_true', default=False, help='Set max # threads')
    parser.add_argument('--num_threads', type=int, dest='num_threads', default=1, help='Number of CPU cores to use')
    args = parser.parse_args()

    os.makedirs(args.out)
    os.makedirs(os.path.join(args.out, "pos"))
    os.makedirs(os.path.join(args.out, "properties"))
    os.makedirs(os.path.join(args.out, "distmat"))
    tinker_md(args)


if __name__ == '__main__':
    main()
