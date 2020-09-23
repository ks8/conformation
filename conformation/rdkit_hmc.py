""" Hamiltonian Monte Carlo conformational search using RDKit. """
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.stats import multivariate_normal
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdForceFieldHelpers, rdMolAlign
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    smiles: str  # Molecular SMILES string
    max_attempts: int = 10000  # Max number of embedding attempts
    temp: float = 298.0  # Temperature for computing Boltzmann probabilities
    epsilon: float = 0.1  # Leapfrog step size
    L: int = 10  # Number of leapfrog steps
    num_steps: int = 1  # Number of HMC steps
    save_dir: str  # Path to directory containing output files


# noinspection PyUnresolvedReferences
def calc_energy_grad(mol: rdchem.Mol) -> Tuple[float, np.ndarray]:
    """
    Compute MMFF energy and gradient.
    :return: MMFF energy and gradient.
    """
    mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    mmff_f = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mmff_p)

    energy = mmff_f.CalcEnergy()
    grad = mmff_f.CalcGrad()
    grad = np.reshape(np.array(grad), [int(len(grad)/3), 3])

    return energy, grad


def calc_energy_grad_normal(q: np.ndarray, cov: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Function for computing
    :param q: Array of points.
    :param cov: Covariance matrix of the potential energy.
    :return:
    """
    cov_inv = np.linalg.inv(cov)
    energy = float(np.matmul(q, np.matmul(cov_inv, q)))
    grad = np.array([0.5*(2*cov_inv[0][0]*q[0] + cov_inv[0][1]*q[1] + cov_inv[1][0]*q[1]),
                     0.5*(cov_inv[0][1]*q[0] + cov_inv[1][0]*q[0] + 2*cov_inv[1][1]*q[1])])
    return energy, grad


def rdkit_hmc(args: Args) -> None:
    """
    Metropolis-Hastings conformational search using RDKit.
    :param args: System arguments.
    :return: None.
    """
    # current_q = np.array([-1.5, -1.55])
    # cov = np.array([[1.0, 0.98], [0.98, 1.0]])
    #
    # positions = [current_q]
    #
    # for i in range(args.num_steps):
    #     # Set the current position variables
    #     q = current_q
    #
    #     # Generate random momentum values
    #     if i == 0:
    #         p = np.array([-1, 1])
    #     else:
    #         p = np.random.multivariate_normal(np.zeros(2), np.identity(2))
    #     current_p = p
    #
    #     # Make a half-step for momentum at the beginning
    #     _, grad_u = calc_energy_grad_normal(q, cov)
    #     p = p - args.epsilon * grad_u/2
    #
    #     # Alternate full steps for position and momentum
    #     for j in range(args.L):
    #         # Make a full step for the position
    #         q = q + args.epsilon * p
    #
    #         # Make a full step for the momentum, except at the end of the trajectory
    #         if j != args.L - 1:
    #             _, grad_u = calc_energy_grad_normal(q, cov)
    #             p = p - args.epsilon * grad_u
    #
    #     # Make a half step for momentum at the end
    #     _, grad_u = calc_energy_grad_normal(q, cov)
    #     p = p - args.epsilon * grad_u/2
    #
    #     # Negate the momentum at the end of the trajectory to make the proposal symmetric
    #     p = -p
    #
    #     # Evaluate potential and kinetic energies at start and end of the trajectory
    #     current_u, _ = calc_energy_grad_normal(current_q, cov)
    #     current_k = np.dot(current_p, current_p)/2.
    #     proposed_u, _ = calc_energy_grad_normal(q, cov)
    #     proposed_k = np.dot(p, p)/2.
    #
    #     prob_ratio = math.exp(current_u - proposed_u + current_k - proposed_k)
    #     mu = random.uniform(0, 1)
    #     if mu <= prob_ratio:
    #         current_q = q
    #     positions.append(current_q)
    #
    # positions = np.array(positions)
    # plt.plot(positions[:, 0], positions[:, 1], 'bo', linestyle='-')
    # plt.xlim((-2.4, 2.4))
    # plt.ylim((-2.4, 2.4))
    # plt.savefig("test_joint")
    # plt.clf()
    # plt.plot(positions[:, 0], 'bo')
    # plt.ylim((-3.1, 3.1))
    # plt.savefig("test_marginal")
    # plt.clf()
    os.makedirs(args.save_dir)

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.MolFromSmiles(args.smiles)
    mol = Chem.AddHs(mol)

    print(f'Starting search: {args.smiles}')

    # Generate initial conformation and minimize it
    # Here, we consider the variables of interest, q, to effectively be the atomic coordinates
    current_q = copy.deepcopy(mol)
    AllChem.EmbedMolecule(current_q, maxAttempts=args.max_attempts)
    AllChem.MMFFOptimizeMoleculeConfs(current_q)
    num_atoms = current_q.GetNumAtoms()

    conformation_molecules = [current_q]

    print(f'Running HMC steps...')
    num_accepted = 0
    for _ in tqdm(range(args.num_steps)):
        # Set the current position variables
        q = copy.deepcopy(current_q)

        # Generate random momentum values
        p = np.array([np.random.multivariate_normal(np.zeros(3), np.identity(3)) for _ in range(num_atoms)])
        current_p = p

        # Make a half-step for momentum at the beginning
        _, grad_u = calc_energy_grad(q)
        p = p - args.epsilon * grad_u/2

        # Alternate full steps for position and momentum
        for i in range(args.L):
            # Make a full step for the position
            c = q.GetConformer()
            pos = c.GetPositions()
            pos_init = pos
            pos = pos + args.epsilon * p
            # print("Change in pos: ")
            # print(pos - pos_init)

            # Save updated atomic coordinates to the conformation object
            for j in range(len(pos)):
                c.SetAtomPosition(j, Point3D(pos[j][0], pos[j][1], pos[j][2]))

            # Make a full step for the momentum, except at the end of the trajectory
            if i != args.L - 1:
                energy, grad_u = calc_energy_grad(q)
                p_init = p
                p = p - args.epsilon * grad_u
                # print("Grad: ")
                # print(grad_u)
                # print("Change in p: ")
                # print(p - p_init)
                print(energy)
                print(grad_u)

        # Make a half step for momentum at the end
        _, grad_u = calc_energy_grad(q)
        p = p - args.epsilon * grad_u/2

        # Negate the momentum at the end of the trajectory to make the proposal symmetric
        p = -p

        # Evaluate potential and kinetic energies at start and end of the trajectory
        current_u, _ = calc_energy_grad(current_q)
        current_k = np.sum(np.sum(np.square(current_p), axis=1)/2.)
        proposed_u, _ = calc_energy_grad(q)
        proposed_k = np.sum(np.sum(np.square(p), axis=1)/2.)

        prob_ratio = math.exp(current_u - proposed_u + current_k - proposed_k)
        mu = random.uniform(0, 1)
        if mu <= prob_ratio:
            current_q = q
            conformation_molecules.append(current_q)
            num_accepted += 1

    print(f'Number of conformations identified: {len(conformation_molecules)}')
    print(f'% Moves accepted: {float(num_accepted) / float(args.num_steps) * 100.0}')

    # Save unique conformations in molecule object
    print(f'Saving conformations...')
    for i in range(len(conformation_molecules)):
        c = conformation_molecules[i].GetConformer()
        c.SetId(i)
        mol.AddConformer(c)

    print(f'Minimizing conformations...')
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    post_minimize_energies = []
    for i in range(len(res)):
        post_minimize_energies.append(res[i][1])

    print(f'RMSD pruning...')
    # List of conformers to remove
    unique_conformer_indices = []
    # noinspection PyPep8Naming
    mol_no_Hs = Chem.RemoveHs(mol)

    # Loop through conformations to find unique ones
    print(f'Begin pruning...')
    for i in tqdm(range(mol.GetNumConformers())):
        unique = True
        for j in unique_conformer_indices:
            # noinspection PyUnboundLocalVariable
            energy_diff = abs(post_minimize_energies[i] - post_minimize_energies[j])
            if energy_diff < 2.0:
                # noinspection PyUnboundLocalVariable
                rmsd = rdMolAlign.AlignMol(mol_no_Hs, mol_no_Hs, j, i)
                if rmsd < 0.05:
                    unique = False
                    break
        if unique:
            unique_conformer_indices.append(i)

    print(f'Number of unique post minimization conformations identified: {len(unique_conformer_indices)}')


