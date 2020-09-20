""" Metropolis-Hastings conformational search using RDKit. """
import copy
import math
import matplotlib.pyplot as plt
import numpy as np 
import os
from typing import List, Tuple

import random
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Chem.Lipinski import RotatableBondSmarts
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
from scipy.stats import truncnorm
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    num_steps: int = 1000  # Number of MC steps to perform
    max_attempts: int = 10000  # Max number of embedding attempts
    temp: float = 298.0  # Temperature for computing Boltzmann probabilities
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values
    e_threshold: float = 50  # Energy cutoff
    num_bonds_to_rotate: int = None  # Number of bonds to rotate in proposal distribution
    random_bonds: bool = False  # Whether or not to select a random number of bonds to rotate in proposal distribution
    minimize: bool = False  # Whether or not to energy-minimize proposed samples
    post_minimize: bool = False  # Whether or not to energy-minimize saved samples after MC
    post_rmsd: bool = False  # Whether to RMSD prune saved (and energy-minimized if post_minimize=True) samples after MC
    post_rmsd_threshold: float = 0.65  # RMSD threshold for post minimized conformations
    post_rmsd_energy_diff: float = 3.0  # Energy difference above which two conformations are assumed to be different
    clip_deviation: float = 2.0  # Distance of clip values for truncated normal on either side of the mean
    trunc_std: float = 1.0  # Standard deviation desired for truncated normal
    random_std: bool = False  # Whether or not to select a random trunc_std value at each MC step
    random_std_range: List = [0.1, 1.5]  # Range for random trunc_std value
    subsample_frequency: int = 1  # Frequency at which configurations are saved from MH steps
    log_frequency: int = 1000  # Log frequency
    save_dir: str  # Path to output file


# noinspection PyUnresolvedReferences
def rotate_bonds(current_sample: Chem.rdchem.Mol, rotatable_bonds: Tuple, args: Args) -> Chem.rdchem.Mol:
    """
    Proposal distribution that rotates a specified number of rotatable bonds by a random amount.
    :param current_sample: Current geometry.
    :param rotatable_bonds: Number of rotatable bonds.
    :param args: System arguments.
    :return:
    """
    # Initialize proposed sample
    proposed_sample = copy.deepcopy(current_sample)
    proposed_conf = proposed_sample.GetConformer()

    # Rotate bond(s) (via dihedral angle) by a random amount
    rotatable_bonds = list(rotatable_bonds)
    if args.random_bonds:
        bonds_to_rotate = random.sample(rotatable_bonds, random.randint(1, len(rotatable_bonds)))
    elif args.num_bonds_to_rotate is not None:
        bonds_to_rotate = random.sample(rotatable_bonds, args.num_bonds_to_rotate)
    else:
        bonds_to_rotate = rotatable_bonds
    for bond in bonds_to_rotate:
        # Get atom indices for the ith bond
        atom_a_idx = bond[0]
        atom_b_idx = bond[1]

        # Select a neighbor for each atom in order to form a dihedral
        atom_a_neighbors = proposed_sample.GetAtomWithIdx(atom_a_idx).GetNeighbors()
        atom_a_neighbor_index = [x.GetIdx() for x in atom_a_neighbors if x.GetIdx() != atom_b_idx][0]
        atom_b_neighbors = proposed_sample.GetAtomWithIdx(atom_b_idx).GetNeighbors()
        atom_b_neighbor_index = [x.GetIdx() for x in atom_b_neighbors if x.GetIdx() != atom_a_idx][0]

        # Compute current dihedral angle
        current_angle = rdMolTransforms.GetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                                       atom_b_idx, atom_b_neighbor_index)

        # Perturb the current angle
        if args.random_std:
            trunc_std = random.uniform(args.random_std_range[0], args.random_std_range[1])
        else:
            trunc_std = args.trunc_std
        new_angle = truncnorm.rvs(-args.clip_deviation, args.clip_deviation, loc=current_angle,
                                  scale=trunc_std)

        # Set the dihedral angle to random angle
        rdMolTransforms.SetDihedralRad(proposed_conf, atom_a_neighbor_index, atom_a_idx,
                                       atom_b_idx, atom_b_neighbor_index, new_angle)

    return proposed_sample


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

    # Molecule conformation list
    conformation_molecules = []
    energies = []
    all_conformation_molecules = []
    all_energies = []

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)

    print(f'Starting search: {args.smiles}')

    # Discover the rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    print(f'Num rotatable bonds: {len(rotatable_bonds)}')

    # Generate initial conformation and minimize it
    current_sample = copy.deepcopy(mol)
    AllChem.EmbedMolecule(current_sample, maxAttempts=args.max_attempts)
    res = AllChem.MMFFOptimizeMoleculeConfs(current_sample)
    current_energy = res[0][1] * 1000.0 / avogadro
    conformation_molecules.append(current_sample)
    energies.append(res[0][1])
    all_conformation_molecules.append(current_sample)
    all_energies.append(res[0][1])

    # Run MC steps
    print(f'Running MC steps...')
    num_accepted = 0
    for step in tqdm(range(args.num_steps)):
        proposed_sample = rotate_bonds(current_sample, rotatable_bonds, args)

        # Compute the energy of the proposed sample
        if args.minimize:
            res = AllChem.MMFFOptimizeMoleculeConfs(proposed_sample)
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

            # Save the proposed sample to the list of conformations
            conformation_molecules.append(proposed_sample)
            energies.append(res[0][1])

            num_accepted += 1

        if step % args.subsample_frequency == 0:
            all_conformation_molecules.append(current_sample)
            all_energies.append(res[0][1])

        if step % args.log_frequency == 0:
            print(f'Steps completed: {step}, Num conformations: {len(conformation_molecules)}')

    print(f'Number of conformations identified: {len(conformation_molecules)}')
    print(f'% Moves accepted: {float(num_accepted)/float(args.num_steps)*100.0}')

    with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
        f.write("Number of rotatable bonds: " + str(len(rotatable_bonds)))
        f.write('\n')
        f.write("Number of unique conformations identified: " + str(len(conformation_molecules)))
        f.write('\n')
        f.write("% Moves accepted: " + str(float(num_accepted)/float(args.num_steps)*100.0))
        f.write('\n')

    # Save unique conformations in molecule object
    print(f'Saving conformations...')
    for i in range(len(conformation_molecules)):
        c = conformation_molecules[i].GetConformer()
        c.SetId(i)
        mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = mol.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)

    # Save all conformations in molecule object
    all_mol = Chem.MolFromSmiles(args.smiles)
    all_mol = Chem.AddHs(all_mol)
    for i in range(len(all_conformation_molecules)):
        c = all_conformation_molecules[i].GetConformer()
        c.SetId(i)
        all_mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = all_mol.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)

    if args.post_minimize:
        print(f'Minimizing conformations...')
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        post_minimize_energies = []
        for i in range(len(res)):
            post_minimize_energies.append(res[i][1])

        # Save molecule to binary file
        bin_str = mol.ToBinary()
        with open(os.path.join(args.save_dir, "post-minimization-conformations.bin"), "wb") as b:
            b.write(bin_str)

    if args.post_rmsd:
        print(f'RMSD pruning...')
        # List of conformers to remove
        unique_conformer_indices = []

        if args.rmsd_remove_Hs:
            # noinspection PyPep8Naming
            mol_no_Hs = Chem.RemoveHs(mol)

        # Loop through conformations to find unique ones
        print(f'Begin pruning...')
        for i in tqdm(range(mol.GetNumConformers())):
            unique = True
            for j in unique_conformer_indices:
                if args.post_minimize:
                    # noinspection PyUnboundLocalVariable
                    energy_diff = abs(post_minimize_energies[i] - post_minimize_energies[j])
                else:
                    energy_diff = abs(energies[i] - energies[j])
                if energy_diff < args.post_rmsd_energy_diff:
                    if args.rmsd_remove_Hs:
                        # noinspection PyUnboundLocalVariable
                        rmsd = rdMolAlign.AlignMol(mol_no_Hs, mol_no_Hs, j, i)
                    else:
                        rmsd = rdMolAlign.AlignMol(mol, mol, j, i)
                    if rmsd < args.post_rmsd_threshold:
                        unique = False
                        break
            if unique:
                unique_conformer_indices.append(i)

        print(f'Number of unique post minimization conformations identified: {len(unique_conformer_indices)}')
        with open(os.path.join(args.save_dir, "info.txt"), "a") as f:
            f.write("Number of unique post rmsd conformations: " + str(len(unique_conformer_indices)))
            f.write('\n')

        # Save unique conformers in molecule object
        print(f'Saving conformations...')
        post_rmsd_mol = copy.deepcopy(mol)
        post_rmsd_mol.RemoveAllConformers()
        count = 0
        for i in unique_conformer_indices:
            c = mol.GetConformer(i)
            c.SetId(count)
            post_rmsd_mol.AddConformer(c)
            count += 1

        # Save molecule to binary file
        bin_str = post_rmsd_mol.ToBinary()
        with open(os.path.join(args.save_dir, "post-rmsd-conformations.bin"), "wb") as b:
            b.write(bin_str)

        # Save pruned energies
        res = AllChem.MMFFOptimizeMoleculeConfs(post_rmsd_mol, maxIters=0)
        post_rmsd_energies = []
        for i in range(len(res)):
            post_rmsd_energies.append(res[i][1])

    print(f'Plotting energy distributions...')
    # Plot energy histograms
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax)
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Density")
    ax.figure.savefig(os.path.join(args.save_dir, "energy-distribution.png"))
    plt.clf()
    plt.close()

    if args.post_minimize:
        # noinspection PyUnboundLocalVariable
        fig, ax = plt.subplots()
        # noinspection PyUnboundLocalVariable
        sns.histplot(post_minimize_energies, ax=ax)
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Frequency")
        ax.figure.savefig(os.path.join(args.save_dir, "post-minimization-energy-distribution.png"))
        plt.clf()
        plt.close()

    if args.post_rmsd:
        # noinspection PyUnboundLocalVariable
        fig, ax = plt.subplots()
        # noinspection PyUnboundLocalVariable
        sns.histplot(post_rmsd_energies, ax=ax, bins=np.arange(min(post_rmsd_energies) - 1., max(post_rmsd_energies) +
                                                               1., 0.1))
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Frequency")
        ax.figure.savefig(os.path.join(args.save_dir, "post-rmsd-energy-distribution.png"))
        plt.clf()
        plt.close()
