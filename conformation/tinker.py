""" Run Tinker MD simulations and extract conformations. """
import numpy as np
import os

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdmolfiles, rdMolTransforms
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
# noinspection PyPackageRequirements
from tap import Tap
from typing import List

from conformation.distance_matrix import dist_matrix


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing binary files
    save_dir: str  # Path to directory containing output files
    param_path: str = "/data/swansonk1/anaconda3/envs/my-rdkit-env/Tinker-FFE/tinker/params/mmff"  # Tinker param file
    integrator: str = "verlet"  # Integrator
    num_steps: int = 100000  # Number of MD steps
    time_step: float = 1.0  # Time step in femtoseconds
    save_step: float = 0.1  # Time btw saves in picoseconds
    ensemble: int = 2  # 1=NVE, 2=NVT
    temp: int = 298  # Temperature in degrees Kelvin
    dihedral: bool = False  # Use when computing dihedral angle values
    dihedral_vals: List[int] = [2, 0, 1, 5]
    num_starts: int = 1  # Number of MD restarts from low energy conformations generated by RDKit
    thread_set: bool = True  # Set max number of threads
    num_threads: int = 1  # Number of CPU cores to use
    max_attempts: int = 10000  # Max attempts for embedding a molecule


def tinker(args: Args) -> None:
    """
    Generate MD simulation and conformations for molecules from SMILES string input.
    :param args: Argparse arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "properties"))
    os.makedirs(os.path.join(args.save_dir, "distmat"))

    # Keep track of any molecules that have issues
    failed_molecules = []

    for _, _, files in os.walk(args.data_dir):
        for f in files:
            # Set molecule, file, and smiles variables
            molecule_name = f[:f.find(".")]
            sdf_name = os.path.join(args.save_dir, molecule_name + "." + "sdf")
            key_name = os.path.join(args.save_dir, molecule_name + "." + "key")

            # Initialize conformation counter
            counter = 0

            try:
                # Generate initial MMFF-minimized RDKit conformations from which to start MD simulations
                # noinspection PyUnresolvedReferences
                m2 = Chem.Mol(open(os.path.join(args.data_dir, f), "rb").read())
                _ = AllChem.EmbedMultipleConfs(m2, numConfs=args.num_starts, maxAttempts=args.max_attempts)
                _ = AllChem.MMFFOptimizeMoleculeConfs(m2)
                print(rdmolfiles.MolToPDBBlock(m2), file=open(molecule_name, "w+"))

                # Create folder to hold distance matrices
                os.makedirs(os.path.join(args.save_dir, "distmat", molecule_name))

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

                # Move the xyz files to the args.save_dir folder
                os.system("mv " + molecule_name + "*." + "xyz " + args.save_dir)

                # Remove extraneous key files
                os.system("rm " + molecule_name + "*." + "key ")

                # Create an overall molecule object for writing generated conformations to binary file
                # noinspection PyUnresolvedReferences
                mol = Chem.Mol(open(os.path.join(args.data_dir, f), "rb").read())
                mol.RemoveAllConformers()  # Clear existing conformers

                # Run MD simulations and conformation extraction for each RDKit initial configuration
                for j in range(args.num_starts):
                    # Run the MD simulation
                    try:
                        os.system("dynamic " + os.path.join(args.save_dir, molecule_name + "_" + ("{:0" +
                                                                                                  str(len(str(
                                                                                                      args.num_starts)))
                                                                                                  + "d}").
                                                            format(j + 1) + ".xyz") + " -k " + key_name + " " +
                                  str(args.num_steps) + " " + str(args.time_step) + " " + str(args.save_step) + " " +
                                  str(args.ensemble) + " " + str(args.temp))

                        # Create a random conformation object, used for computing properties
                        # noinspection PyUnresolvedReferences
                        m2 = Chem.Mol(open(os.path.join(args.data_dir, f), "rb").read())
                        _ = AllChem.EmbedMultipleConfs(m2, numConfs=1, maxAttempts=10*args.max_attempts)
                        c = m2.GetConformers()[0]

                        # Open the trajectory (.arc) file and process the conformations
                        with open(
                                os.path.join(args.save_dir, molecule_name + "_" + ("{:0" +
                                                                                   str(len(str(args.num_starts))) +
                                                                                   "d}").format(j + 1) + ".arc")) as \
                                tmp:
                            line = tmp.readline()
                            pos = []
                            new_file = True
                            while line:
                                # If we have finished extracting coordinates for one conformation, process conformation
                                if line.split()[1] == molecule_name:
                                    if not new_file:

                                        # Keep track of conformer ID when adding to overall molecule object
                                        c.SetId(counter)

                                        # Save the atomic positions as a numpy array
                                        pos = np.array(pos)

                                        # Compute pairwise distance matrix and save to a numpy file in "distmat" folder
                                        dist_matrix(pos,
                                                    os.path.join(args.save_dir, "distmat", molecule_name, "distmat-" +
                                                                 str(counter) + "-" + molecule_name))

                                        # Save atomic coordinates to the conformation object
                                        for i in range(len(pos)):
                                            c.SetAtomPosition(i, Point3D(pos[i][0], pos[i][1], pos[i][2]))

                                        # Add the conformer to the overall molecule object
                                        mol.AddConformer(c)

                                        # Compute the specified dihedral angle
                                        if args.dihedral:
                                            dihedral = rdMolTransforms.GetDihedralRad(c, args.dihedral_vals[0],
                                                                                      args.dihedral_vals[1],
                                                                                      args.dihedral_vals[2],
                                                                                      args.dihedral_vals[3])
                                        else:
                                            dihedral = "nan"

                                        # Compute the potential energy of the conformation
                                        res = AllChem.MMFFOptimizeMoleculeConfs(m2, maxIters=0)

                                        # Write property information to a text file in the "properties" folder
                                        file_name = "energy-rms-dihedral-" + str(counter) + "-" + molecule_name + ".txt"
                                        with open(os.path.join(args.save_dir, "properties", file_name), "w") as o:
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
                    except FileNotFoundError:
                        continue

                # Print the conformations to a binary file
                bin_str = mol.ToBinary()
                with open(os.path.join(args.save_dir, molecule_name + "-conformations.bin"), "wb") as b:
                    b.write(bin_str)

            except ValueError:
                failed_molecules.append(f)
                continue
