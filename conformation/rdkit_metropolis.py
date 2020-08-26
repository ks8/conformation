""" Metropolis-Hastings conformational search using RDKit. """
import copy
import math
import os

import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing RDKit mol binaries
    save_dir: str  # Path to directory for saving output files
    num_steps: int = 1000  # Number of MC steps to perform
    max_attempts: int = 10000  # Max number of embedding attempts
    temp: float = 298.0  # Temperature for computing Boltzmann probabilities
    rmsd_threshold: float = 1.0  # RMSD threshold for determining identical conformations
    log_frequency: int = 1000  # Log frequency


def rdkit_metropolis(args: Args) -> None:
    """
    Metropolis-Hastings conformational search using RDKit.
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    avogadro = 6.022e23

    # Loop through all binary molecule files
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            # Molecule conformation list
            conformation_molecules = []

            # Load molecule
            # noinspection PyUnresolvedReferences
            mol = Chem.Mol(open(os.path.join(args.data_dir, f), "rb").read())

            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            print(f'Starting Search: {smiles}')

            # Discover the rotatable bonds
            rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
            print(f'Num Rotatable Bonds: {len(rotatable_bonds)}')

            # Generate initial conformation and minimize it
            current_sample = copy.deepcopy(mol)
            AllChem.EmbedMolecule(current_sample, maxAttempts=args.max_attempts)
            res = AllChem.MMFFOptimizeMoleculeConfs(current_sample)
            current_energy = res[0][1]*1000.0/avogadro
            conformation_molecules.append(current_sample)

            # Run MC steps
            for step in range(args.num_steps):
                # Initialize proposed sample
                proposed_sample = copy.deepcopy(current_sample)
                proposed_conf = proposed_sample.GetConformer()

                # Randomly select a random number of rotatable bonds to rotate
                selected_bonds = random.sample(rotatable_bonds, 1)

                # Rotate each of these bonds (via dihedral angle) a uniformly random amount
                for i in range(len(selected_bonds)):
                    # Get atom indices for the ith bond
                    atom_a_idx = selected_bonds[i][0]
                    atom_b_idx = selected_bonds[i][1]

                    # Select a neighbor for each atom in order to form a dihedral
                    atom_a_neighbors = proposed_sample.GetAtomWithIdx(atom_a_idx).GetNeighbors()
                    atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
                    atom_b_neighbors = proposed_sample.GetAtomWithIdx(atom_b_idx).GetNeighbors()
                    atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

                    # Compute the current dihedral angle in radians
                    cur_angle = rdMolTransforms.GetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                                               atom_b_idx, atom_b_neighbor_index)

                    # Randomly select a new angle
                    new_angle = cur_angle + random.uniform(-math.pi, math.pi)

                    # Set the dihedral angle to the new angle
                    rdMolTransforms.SetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                                   atom_b_idx, atom_b_neighbor_index, new_angle)

                # Compute the energy of the proposed sample
                res = AllChem.MMFFOptimizeMoleculeConfs(proposed_sample)
                proposed_energy = res[0][1]*1000.0/avogadro

                # Probability ratio
                prob_ratio = math.exp((current_energy - proposed_energy)/(k_b*args.temp))
                mu = random.uniform(0, 1)
                if mu <= prob_ratio:
                    # Update the energy of the current sample to that of the proposed sample
                    current_sample = proposed_sample
                    current_energy = proposed_energy

                    # Save the proposed sample to the list of conformations if it is unique
                    unique = True
                    for i in range(len(conformation_molecules)):
                        rmsd = rdMolAlign.AlignMol(conformation_molecules[i], proposed_sample)
                        if rmsd < args.rmsd_threshold:
                            unique = False
                            break
                    if unique:
                        conformation_molecules.append(proposed_sample)

                if step % args.log_frequency == 0:
                    print(f'Steps Completed: {step}, Num Conformations: {len(conformation_molecules)}')

            print(f'Number of Unique Conformations Identified: {len(conformation_molecules)}')

            # Save unique conformers in molecule object
            for i in range(len(conformation_molecules)):
                c = conformation_molecules[i].GetConformer()
                c.SetId(i)
                mol.AddConformer(c)

            # Save molecule to binary file
            bin_str = mol.ToBinary()
            with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
                b.write(bin_str)
