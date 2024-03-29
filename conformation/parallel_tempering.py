""" General framework for parallel tempering MCMC. """
import copy
from logging import Logger
import math
import numpy as np
import os
import random
import time
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers
from rdkit.Chem.Lipinski import RotatableBondSmarts
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm

from conformation.rdkit_hmc import calc_energy_grad, hmc_step


class Args(Tap):
    """
    System arguments.
    """
    bin_path: str  # Path to RDKit binary file containing molecule
    max_attempts: int = 10000  # Max number of embedding attempts
    num_minimization_iters: int = 0  # Number of minimization steps
    temperatures: List[float] = [300, 500]  # Temperature ladder, with the first being the target temperature
    epsilon: float = 1  # Leapfrog step size in femtoseconds
    L: int = 10  # Number of leapfrog steps
    num_steps: int = 1  # Number of parallel tempering steps
    swap_prob: float = 0.2  # Probability of performing a swap operation on a given step
    subsample_frequency: int = 1  # Frequency at which configurations are saved from MH steps
    log_frequency: int = 100  # Log frequency
    save_dir: str  # Path to directory containing output files


def parallel_tempering(args: Args, logger: Logger) -> None:
    """
    Parallel tempering scheme.
    :param args: System arguments.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    avogadro = 6.022e23
    args.epsilon *= 1e-15  # Convert to femtoseconds

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.bin_path, "rb").read())
    mol.RemoveAllConformers()

    debug(f'Starting search...')

    # Generate initial conformations
    # Here, we consider the variables of interest, q, to effectively be the atomic coordinates
    initial_q = copy.deepcopy(mol)
    num_atoms = initial_q.GetNumAtoms()
    AllChem.EmbedMultipleConfs(initial_q, maxAttempts=args.max_attempts, numConfs=len(args.temperatures), numThreads=0)
    AllChem.MMFFOptimizeMoleculeConfs(initial_q, maxIters=args.num_minimization_iters, numThreads=0)
    current_q_list = []
    for i in range(len(args.temperatures)):
        current_q = copy.deepcopy(mol)
        c = initial_q.GetConformers()[i]
        c.SetId(0)
        current_q.AddConformer(c)
        current_q_list.append(current_q)

    # Compute force field information
    mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(initial_q)
    force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(initial_q, mmff_p)

    # Masses in kg
    mass = np.array([mol.GetAtomWithIdx(i).GetMass() / (1000. * avogadro) for i in range(num_atoms)])

    # Add the first conformation to the list
    energy, _ = calc_energy_grad(current_q_list[0].GetConformer().GetPositions(), force_field)
    all_conformation_molecules = [current_q_list[0]]

    debug(f'Running HMC steps...')
    start_time = time.time()
    num_internal_accepted = [0]*len(args.temperatures)
    num_swap_accepted = [0]*(len(args.temperatures) - 1)
    num_swap_attempted = [0]*(len(args.temperatures) - 1)
    total_num_swap_accepted = 0
    total_swap_attempted = 0
    for step in tqdm(range(args.num_steps)):
        alpha = random.uniform(0, 1)
        if alpha > args.swap_prob:
            results = []
            for i in range(len(args.temperatures)):
                accepted, current_q, current_energy = hmc_step(current_q_list[i], force_field, args.temperatures[i],
                                                               k_b, avogadro, mass, num_atoms, args.epsilon, args.L)
                results.append([accepted, current_q, current_energy])

            for i in range(len(args.temperatures)):
                accepted, current_q, current_energy = results[i]

                if accepted:
                    current_q_list[i] = current_q  # Necessary because Python is pass by object reference!
                    num_internal_accepted[i] += 1

                if i == 0:
                    if step % args.subsample_frequency == 0:
                        all_conformation_molecules.append(current_q)

        elif len(args.temperatures) > 1:
            swap_index = random.randint(0, len(args.temperatures) - 2)
            energy_k0, _ = calc_energy_grad(current_q_list[swap_index].GetConformer().GetPositions(), force_field)
            energy_k0 *= (1000.0 * 4.184 / avogadro)
            energy_k1, _ = calc_energy_grad(current_q_list[swap_index + 1].GetConformer().GetPositions(), force_field)
            energy_k1 *= (1000.0 * 4.184 / avogadro)
            delta_beta = (1./(k_b * args.temperatures[swap_index] * 4.184) -
                          1./(k_b * args.temperatures[swap_index + 1] * 4.184))
            delta_energy = energy_k0 - energy_k1
            delta = delta_beta * delta_energy

            prob_ratio = math.exp(-delta)
            mu = random.uniform(0, 1)
            if swap_index == 0:
                energy, _ = calc_energy_grad(current_q_list[0].GetConformer().GetPositions(), force_field)
            if mu <= prob_ratio:
                tmp = copy.deepcopy(current_q_list[swap_index])
                current_q_list[swap_index] = copy.deepcopy(current_q_list[swap_index + 1])
                current_q_list[swap_index + 1] = tmp
                num_swap_accepted[swap_index] += 1
                total_num_swap_accepted += 1
            if swap_index == 0:
                if step % args.subsample_frequency == 0:
                    all_conformation_molecules.append(current_q_list[0])
            num_swap_attempted[swap_index] += 1
            total_swap_attempted += 1

        if step % args.log_frequency == 0:
            for i in range(len(args.temperatures)):
                debug(f'% Moves accepted for temperature {args.temperatures[i]}: '
                      f'{float(num_internal_accepted[i]) / float(step + 1) * 100.0}')

            for i in range(len(args.temperatures) - 1):
                if num_swap_attempted[i] == 0:
                    debug(f'% Moves accepted for swap at base temperature {args.temperatures[i]}: NA')
                else:
                    debug(f'% Moves accepted for swap at base temperature {args.temperatures[i]}: '
                          f'{float(num_swap_accepted[i]) / num_swap_attempted[i] * 100.0}')

            debug(f'# Swap moves attempted: {total_swap_attempted}')
            if total_swap_attempted == 0:
                debug(f'% Moves accepted for swap: {0.0}')
                debug(f'% Moves accepted for swap: NA')
            else:
                debug(f'% Moves accepted for swap: '
                      f'{float(total_num_swap_accepted) / float(total_swap_attempted) * 100.0}')
    end_time = time.time()
    debug(f'Total Time (s): {end_time - start_time}')
    for i in range(len(args.temperatures)):
        debug(f'% Moves accepted for temperature {args.temperatures[i]}: '
              f'{float(num_internal_accepted[i]) / float(args.num_steps) * 100.0}')
    debug(f'# Swap moves attempted: {total_swap_attempted}')
    if total_swap_attempted > 0:
        debug(f'% Moves accepted for swap: {float(total_num_swap_accepted) / float(total_swap_attempted) * 100.0}')

    # Discover the rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    debug(f'Num rotatable bonds: {len(rotatable_bonds)}')

    # Save all sub sampled conformations in molecule object
    # noinspection PyUnresolvedReferences
    all_mol = Chem.Mol(open(args.bin_path, "rb").read())
    all_mol.RemoveAllConformers()
    for i in range(len(all_conformation_molecules)):
        c = all_conformation_molecules[i].GetConformer()
        c.SetId(i)
        all_mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = all_mol.ToBinary()
    with open(os.path.join(args.save_dir, "conformations.bin"), "wb") as b:
        b.write(bin_str)
