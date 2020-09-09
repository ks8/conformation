""" Metropolis-Hastings conformational search using RDKit. """
import copy
import math
import matplotlib.pyplot as plt

import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
from scipy.stats import multivariate_normal, truncnorm
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    save_path: str  # Path to output file
    num_steps: int = 1000  # Number of MC steps to perform
    max_attempts: int = 10000  # Max number of embedding attempts
    temp: float = 298.0  # Temperature for computing Boltzmann probabilities
    rmsd_threshold: float = 0.65  # RMSD threshold for determining identical conformations
    post_rmsd_threshold: float = 0.65  # RMSD threshold for post minimized conformations
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values
    e_threshold: float = 50  # Energy cutoff
    minimize: bool = False  # Whether or not to energy-minimize proposed samples
    post_rmsd: bool = False  # Whether or not to energy-minimize saved samples after MC
    cartesian_coords: bool = False  # Whether or not to use internal coordinates or Cartesian coordinates
    clip_deviation: float = 2.0  # Distance of clip values for truncated normal on either side of the mean
    trunc_std: float = 1.0  # Standard deviation desired for truncated normal
    log_frequency: int = 1000  # Log frequency


def rdkit_metropolis(args: Args) -> None:
    """
    Metropolis-Hastings conformational search using RDKit.
    :param args: System arguments.
    :return: None.
    """
    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    avogadro = 6.022e23

    # Molecule conformation list
    conformation_molecules = []
    energies = []

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()

    print(f'Starting Search: {args.smiles}')

    # Discover the rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    print(f'Num Rotatable Bonds: {len(rotatable_bonds)}')

    # Generate initial conformation and minimize it
    current_sample = copy.deepcopy(mol)
    AllChem.EmbedMolecule(current_sample, maxAttempts=args.max_attempts)
    res = AllChem.MMFFOptimizeMoleculeConfs(current_sample, numThreads=0)
    current_energy = res[0][1] * 1000.0 / avogadro
    conformation_molecules.append(current_sample)
    energies.append(res[0][1])
    lowest_energy = res[0][1]

    # Run MC steps
    num_accepted = 0
    for step in range(args.num_steps):
        # Initialize proposed sample
        proposed_sample = copy.deepcopy(current_sample)
        proposed_conf = proposed_sample.GetConformer()

        if not args.cartesian_coords:
            # Rotate each of these bonds (via dihedral angle) by a random amount
            for i in range(len(rotatable_bonds)):
                # Get atom indices for the ith bond
                atom_a_idx = rotatable_bonds[i][0]
                atom_b_idx = rotatable_bonds[i][1]

                # Select a neighbor for each atom in order to form a dihedral
                atom_a_neighbors = proposed_sample.GetAtomWithIdx(atom_a_idx).GetNeighbors()
                atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
                atom_b_neighbors = proposed_sample.GetAtomWithIdx(atom_b_idx).GetNeighbors()
                atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

                # Compute current dihedral angle
                current_angle = rdMolTransforms.GetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                                               atom_b_idx, atom_b_neighbor_index)

                # Perturb the current angle
                new_angle = truncnorm.rvs(-args.clip_deviation, args.clip_deviation, loc=current_angle,
                                          scale=args.trunc_std)

                # Set the dihedral angle to random angle
                rdMolTransforms.SetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                               atom_b_idx, atom_b_neighbor_index, new_angle)

        else:
            # Move each atom by a random amount
            pos = proposed_conf.GetPositions()
            for i in range(num_atoms):
                current_position = pos[i, :]
                new_coords = multivariate_normal.rvs(mean=[current_position[0], current_position[1],
                                                           current_position[2]], cov=[args.trunc_std]*3)
                new_x, new_y, new_z = new_coords[0], new_coords[1], new_coords[2]
                proposed_conf.SetAtomPosition(i, Point3D(new_x, new_y, new_z))

        # Compute the energy of the proposed sample
        if args.minimize:
            res = AllChem.MMFFOptimizeMoleculeConfs(proposed_sample, numThreads=0)
        else:
            res = AllChem.MMFFOptimizeMoleculeConfs(proposed_sample, maxIters=0)
        proposed_energy = res[0][1] * 1000.0 / avogadro

        # Probability ratio
        prob_ratio = math.exp((current_energy - proposed_energy) / (k_b * args.temp))
        mu = random.uniform(0, 1)
        if mu <= prob_ratio:
            # Update the energy of the current sample to that of the proposed sample
            current_sample = proposed_sample
            current_energy = proposed_energy

            # Save the proposed sample to the list of conformations if it is unique
            unique = True
            if args.rmsd_threshold > 0.0:
                for i in range(len(conformation_molecules)):
                    if args.rmsd_remove_Hs:
                        rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(conformation_molecules[i]),
                                                     Chem.RemoveHs(proposed_sample))
                    else:
                        rmsd = rdMolAlign.GetBestRMS(conformation_molecules[i], proposed_sample)
                    if rmsd < args.rmsd_threshold or res[0][1] - lowest_energy > args.e_threshold:
                        unique = False
                        break
            if unique:
                conformation_molecules.append(proposed_sample)
                energies.append(res[0][1])
                if res[0][1] < lowest_energy:
                    lowest_energy = res[0][1]

            num_accepted += 1

        if step % args.log_frequency == 0:
            print(f'Steps completed: {step}, Num conformations: {len(conformation_molecules)}')

    print(f'Number of unique conformations identified: {len(conformation_molecules)}')
    print(f'% Moves accepted: {float(num_accepted)/float(args.num_steps)}')

    with open(args.save_path + "-info.txt", "w") as f:
        f.write("Number of rotatable bonds: " + str(len(rotatable_bonds)))
        f.write('\n')
        f.write("Number of unique conformations identified: " + str(len(conformation_molecules)))
        f.write('\n')
        f.write("% Moves accepted: " + str(float(num_accepted)/float(args.num_steps)))
        f.write('\n')

    # Save unique conformers in molecule object
    for i in range(len(conformation_molecules)):
        c = conformation_molecules[i].GetConformer()
        c.SetId(i)
        mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = mol.ToBinary()
    with open(args.save_path + "-conformations.bin", "wb") as b:
        b.write(bin_str)

    if not args.minimize:
        print(f'Minimizing conformations...')
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        post_minimize_energies = []
        for i in range(len(res)):
            post_minimize_energies.append(res[i][1])

    if args.post_rmsd:
        print(f'RMSD pruning...')
        # List of unique post-minimization molecules
        post_conformation_molecules = []

        # Add an initial molecule to the list
        post_mol = copy.deepcopy(mol)
        post_mol.RemoveAllConformers()
        c = mol.GetConformers()[0]
        c.SetId(0)
        post_mol.AddConformer(c)
        post_conformation_molecules.append(post_mol)

        # Loop through the remaining conformations to find unique ones
        for i in range(1, mol.GetNumConformers()):
            # Create a molecule with the current conformation we are checking for uniqueness
            post_mol = copy.deepcopy(mol)
            post_mol.RemoveAllConformers()
            c = mol.GetConformers()[i]
            c.SetId(0)
            post_mol.AddConformer(c)
            unique = True
            for j in range(len(post_conformation_molecules)):
                # Check for uniqueness
                if args.rmsd_remove_Hs:
                    rmsd = rdMolAlign.GetBestRMS(Chem.RemoveHs(post_conformation_molecules[j]), Chem.RemoveHs(post_mol))
                else:
                    rmsd = rdMolAlign.GetBestRMS(post_conformation_molecules[j], post_mol)
                if rmsd < args.post_rmsd_threshold:
                    unique = False
                    break

            if unique:
                post_conformation_molecules.append(post_mol)

        print(f'Number of unique post minimization conformations identified: {len(post_conformation_molecules)}')
        with open(args.save_path + "-info.txt", "a") as f:
            f.write("Number of unique post minimization conformations: " + str(len(post_conformation_molecules)))
            f.write('\n')

        # Save unique conformers in molecule object
        post_mol = copy.deepcopy(mol)
        post_mol.RemoveAllConformers()
        for i in range(len(post_conformation_molecules)):
            c = post_conformation_molecules[i].GetConformer()
            c.SetId(i)
            post_mol.AddConformer(c)

        # Save molecule to binary file
        bin_str = post_mol.ToBinary()
        with open(args.save_path + "-post-minimization-rmsd-conformations.bin", "wb") as b:
            b.write(bin_str)

    # Plot energy distributions
    fig, ax = plt.subplots()
    sns.distplot(energies, ax=ax)
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Density")
    ax.figure.savefig(args.save_path + "-energy-distribution.png")
    plt.clf()
    plt.close()

    if not args.minimize:
        fig, ax = plt.subplots()
        # noinspection PyUnboundLocalVariable
        sns.distplot(post_minimize_energies, ax=ax)
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Density")
        ax.figure.savefig(args.save_path + "-post-minimized-energy-distribution.png")
        plt.clf()
        plt.close()
