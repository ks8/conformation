""" General framework for parallel tempering MCMC. """
import copy
from functools import partial
import math
import multiprocessing as mp
import numpy as np
import os
import random
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem
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
    swap_prob: float = 0.2  # Probability of performing a swap operation on a given sep
    subsample_frequency: int = 1  # Frequency at which configurations are saved from MH steps
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
    args.epsilon *= 1e-15

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
    num_swap_accepted = 0
    total_swap_attempted = 0
    for step in tqdm(range(args.num_steps)):
        pool = mp.Pool(mp.cpu_count())
        results = []

        def callback_function(result, index):
            """
            test.
            :param result:
            :param index:
            :return:
            """
            results.append(list(result) + [index])

        for i in range(len(args.temperatures)):
            new_callback_function = partial(callback_function, index=i)
            pool.apply_async(hmc_step, args=(current_q_list[i], args.temperatures[i], k_b, avogadro, mass, num_atoms,
                                             args.epsilon, args.L), callback=new_callback_function)
        pool.close()
        pool.join()
        results.sort(key=lambda x: x[3])

        for i in range(len(args.temperatures)):
            accepted, current_q, current_energy, _ = results[i]

            if accepted:
                if i == 0:
                    conformation_molecules.append(current_q)
                    energies.append(current_energy)
                num_internal_accepted[i] += 1

            if i == 0:
                if step % args.subsample_frequency == 0:
                    all_conformation_molecules.append(current_q)
                    all_energies.append(current_energy)

        alpha = random.uniform(0, 1)
        if alpha <= args.swap_prob:
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
            if mu <= prob_ratio:
                tmp = current_q_list[swap_index]
                current_q_list[swap_index] = current_q_list[swap_index + 1]
                current_q_list[swap_index + 1] = tmp
                num_swap_accepted += 1
            total_swap_attempted += 1

    print(f'Number of conformations identified: {len(conformation_molecules)}')
    for i in range(len(args.temperatures)):
        print(f'% Moves accepted for temperature {args.temperatures[i]}: '
              f'{float(num_internal_accepted[i]) / float(args.num_steps) * 100.0}')
    print(f'% Moves accepted for swap: {float(num_swap_accepted) / float(total_swap_attempted) * 100.0}')
