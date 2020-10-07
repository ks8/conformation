""" General framework for parallel tempering MCMC. """
import copy
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import random
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Lipinski import RotatableBondSmarts
import seaborn as sns
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.rdkit_hmc import calc_energy_grad, hmc_step


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    max_attempts: int = 10000  # Max number of embedding attempts
    temperatures: List[float] = [300, 500]  # Temperature ladder, with the first being the target temperature
    epsilon: float = 1  # Leapfrog step size in femtoseconds
    L: int = 10  # Number of leapfrog steps
    num_steps: int = 1  # Number of parallel tempering steps
    swap_prob: float = 0.2  # Probability of performing a swap operation on a given step
    post_minimize: bool = False  # Whether or not to energy-minimize saved samples after MC
    post_rmsd: bool = False  # Whether to RMSD prune saved (and energy-minimized if post_minimize=True) samples after MC
    rmsd_remove_Hs: bool = False  # Whether or not to remove Hydrogen when computing RMSD values
    post_rmsd_energy_diff: float = 2.0  # Energy difference above which two conformations are assumed to be different
    post_rmsd_threshold: float = 0.05  # RMSD threshold for post minimized conformations
    subsample_frequency: int = 1  # Frequency at which configurations are saved from MH steps
    parallel: bool = False  # Whether or not to run the program in parallel fashion
    log_frequency: int = 100  # Log frequency
    save_dir: str  # Path to directory containing output files


def parallel_tempering(args: Args) -> None:
    """
    Parallel tempering scheme.
    :param args: System arguments.
    :return: None.
    """
    os.makedirs(args.save_dir)

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    avogadro = 6.022e23
    args.epsilon *= 1e-15  # Convert to femtoseconds

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)

    print(f'Starting search: {args.smiles}')

    # Generate initial conformations
    # Here, we consider the variables of interest, q, to effectively be the atomic coordinates
    initial_q = copy.deepcopy(mol)
    num_atoms = initial_q.GetNumAtoms()
    AllChem.EmbedMultipleConfs(initial_q, maxAttempts=args.max_attempts, numConfs=len(args.temperatures), numThreads=0)
    current_q_list = []
    for i in range(len(args.temperatures)):
        current_q = copy.deepcopy(mol)
        c = initial_q.GetConformers()[i]
        c.SetId(0)
        current_q.AddConformer(c)
        current_q_list.append(current_q)

    # Masses in kg
    mass = np.array([mol.GetAtomWithIdx(i).GetMass() / (1000. * avogadro) for i in range(num_atoms)])

    # Add the first conformation to the list
    energy, _ = calc_energy_grad(current_q_list[0])
    conformation_molecules = [current_q_list[0]]
    energies = [energy]
    all_conformation_molecules = [current_q_list[0]]
    all_energies = [energy]

    print(f'Running HMC steps...')
    num_internal_accepted = [0]*len(args.temperatures)
    num_swap_accepted = [0]*(len(args.temperatures) - 1)
    num_swap_attempted = [0]*(len(args.temperatures) - 1)
    total_num_swap_accepted = 0
    total_swap_attempted = 0
    for step in tqdm(range(args.num_steps)):
        alpha = random.uniform(0, 1)
        if alpha > args.swap_prob:
            if args.parallel:  # Parallelize the MD update across all of the temperatures
                pool = mp.Pool(mp.cpu_count())
                args_list = []
                for i in range(len(args.temperatures)):
                    args_list.append([current_q_list[i], args.temperatures[i], k_b, avogadro, mass,
                                      num_atoms, args.epsilon, args.L])
                results = pool.starmap_async(hmc_step, args_list).get()
                pool.close()
                pool.join()

            else:
                results = []
                for i in range(len(args.temperatures)):
                    accepted, current_q, current_energy = hmc_step(current_q_list[i], args.temperatures[i], k_b,
                                                                   avogadro, mass, num_atoms, args.epsilon, args.L)
                    results.append([accepted, current_q, current_energy])

            for i in range(len(args.temperatures)):
                accepted, current_q, current_energy = results[i]

                if accepted:
                    if i == 0:
                        conformation_molecules.append(current_q)
                        energies.append(current_energy)
                    num_internal_accepted[i] += 1

                if i == 0:
                    if step % args.subsample_frequency == 0:
                        all_conformation_molecules.append(current_q)
                        all_energies.append(current_energy)

        else:
            swap_index = random.randint(0, len(args.temperatures) - 2)
            energy_k0, _ = calc_energy_grad(current_q_list[swap_index])
            energy_k0 *= (1000.0 * 4.184 / avogadro)
            energy_k1, _ = calc_energy_grad(current_q_list[swap_index + 1])
            energy_k1 *= (1000.0 * 4.184 / avogadro)
            delta_beta = (1./(k_b * args.temperatures[swap_index] * 4.184) -
                          1./(k_b * args.temperatures[swap_index + 1] * 4.184))
            delta_energy = energy_k0 - energy_k1
            delta = delta_beta * delta_energy

            prob_ratio = math.exp(-delta)
            mu = random.uniform(0, 1)
            if swap_index == 0:
                energy, _ = calc_energy_grad(current_q_list[0])
            if mu <= prob_ratio:
                tmp = current_q_list[swap_index]
                current_q_list[swap_index] = current_q_list[swap_index + 1]
                current_q_list[swap_index + 1] = tmp
                num_swap_accepted[swap_index] += 1
                total_num_swap_accepted += 1
                if swap_index == 0:
                    conformation_molecules.append(current_q_list[0])
                    energies.append(energy)
            if swap_index == 0:
                if step % args.subsample_frequency == 0:
                    all_conformation_molecules.append(current_q_list[0])
                    all_energies.append(energy)
            num_swap_attempted[swap_index] += 1
            total_swap_attempted += 1

        if step % args.log_frequency == 0:
            print(f'Number of conformations identified: {len(conformation_molecules)}')
            for i in range(len(args.temperatures)):
                print(f'% Moves accepted for temperature {args.temperatures[i]}: '
                      f'{float(num_internal_accepted[i]) / float(step + 1) * 100.0}')

            for i in range(len(args.temperatures) - 1):
                if num_swap_attempted[i] == 0:
                    print(f'% Moves accepted for swap at base temperature {args.temperatures[i]}: NA')
                else:
                    print(f'% Moves accepted for swap at base temperature {args.temperatures[i]}: '
                          f'{float(num_swap_accepted[i]) / num_swap_attempted[i] * 100.0}')

            print(f'# Swap moves attempted: {total_swap_attempted}')
            if total_swap_attempted == 0:
                print(f'% Moves accepted for swap: {0.0}')
                print(f'% Moves accepted for swap: NA')
            else:
                print(f'% Moves accepted for swap: '
                      f'{float(total_num_swap_accepted) / float(total_swap_attempted) * 100.0}')

    print(f'Number of conformations identified: {len(conformation_molecules)}')
    for i in range(len(args.temperatures)):
        print(f'% Moves accepted for temperature {args.temperatures[i]}: '
              f'{float(num_internal_accepted[i]) / float(args.num_steps) * 100.0}')
    print(f'# Swap moves attempted: {total_swap_attempted}')
    print(f'% Moves accepted for swap: {float(total_num_swap_accepted) / float(total_swap_attempted) * 100.0}')

    # Discover the rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    print(f'Num rotatable bonds: {len(rotatable_bonds)}')

    with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
        f.write("Number of rotatable bonds: " + str(len(rotatable_bonds)))
        f.write('\n')
        f.write("Number of conformations accepted: " + str(len(conformation_molecules)))
        f.write('\n')
        for i in range(len(args.temperatures)):
            f.write(f'% Moves accepted for temperature {args.temperatures[i]}: '
                    f'{float(num_internal_accepted[i]) / float(args.num_steps) * 100.0}')
            f.write('\n')
        f.write(f'# Swap moves attempted: {total_swap_attempted}')
        f.write('\n')
        f.write(f'% Moves accepted for swap: {float(total_num_swap_accepted) / float(total_swap_attempted) * 100.0}')
        f.write('\n')

    # Save accepted conformations in molecule object
    print(f'Saving conformations...')
    for i in range(len(conformation_molecules)):
        c = conformation_molecules[i].GetConformer()
        c.SetId(i)
        mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = mol.ToBinary()
    with open(os.path.join(args.save_dir, "accepted-conformations.bin"), "wb") as b:
        b.write(bin_str)

    # Save all sub sampled conformations in molecule object
    all_mol = Chem.MolFromSmiles(args.smiles)
    all_mol = Chem.AddHs(all_mol)
    for i in range(len(all_conformation_molecules)):
        c = all_conformation_molecules[i].GetConformer()
        c.SetId(i)
        all_mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = all_mol.ToBinary()
    with open(os.path.join(args.save_dir, "all-conformations.bin"), "wb") as b:
        b.write(bin_str)

    if args.post_minimize:
        print(f'Minimizing accepted conformations...')
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
        for i in tqdm(range(mol.GetNumConformers())):
            unique = True
            for j in unique_conformer_indices:
                # noinspection PyUnboundLocalVariable
                energy_diff = abs(post_minimize_energies[i] - post_minimize_energies[j])
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

        print(f'Number of unique conformations identified: {len(unique_conformer_indices)}')
        with open(os.path.join(args.save_dir, "info.txt"), "a") as f:
            f.write("Number of unique post rmsd conformations identified: " + str(len(unique_conformer_indices)))
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
    # NOTE: defining the bins is useful because sometimes automatic bin placement takes forever
    fig, ax = plt.subplots()
    sns.histplot(energies, ax=ax, bins=np.arange(min(energies) - 1., max(energies) + 1., 0.1))
    ax.set_xlabel("Energy (kcal/mol)")
    ax.set_ylabel("Frequency")
    ax.figure.savefig(os.path.join(args.save_dir, "energy-distribution.png"))
    plt.clf()
    plt.close()

    if args.post_minimize:
        # noinspection PyUnboundLocalVariable
        fig, ax = plt.subplots()
        # noinspection PyUnboundLocalVariable
        sns.histplot(post_minimize_energies, ax=ax, bins=np.arange(min(post_minimize_energies) - 1.,
                                                                   max(post_minimize_energies) + 1., 0.1))
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

    plt.plot(all_energies)
    plt.savefig(os.path.join(args.save_dir, "all-energies.png"))
