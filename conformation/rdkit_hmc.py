""" Hamiltonian Monte Carlo conformational search using RDKit. """
import copy
import math
import numpy as np
import os
import random
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdForceFieldHelpers, rdMolAlign, rdmolfiles
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
    epsilon: float = 1  # Leapfrog step size in femtoseconds
    L: int = 10  # Number of leapfrog steps
    num_steps: int = 1  # Number of HMC steps
    save_dir: str  # Path to directory containing output files


# noinspection PyUnresolvedReferences
def calc_energy_grad(mol: rdchem.Mol) -> Tuple[float, np.ndarray]:
    """
    Compute MMFF energy and gradient.
    :return: MMFF energy and gradient, where the energy is kcal/mol and the gradient is kcal/mol/Angstrom.
    """
    mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    mmff_f = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mmff_p)

    energy = mmff_f.CalcEnergy()
    grad = mmff_f.CalcGrad()
    grad = np.reshape(np.array(grad), [int(len(grad)/3), 3])

    return energy, grad


def maxwell_boltzmann(temp: float, k_b: float, mass: float) -> np.ndarray:
    """
    Random draw from a Maxwell-Boltzmann distribution for a given temperature and mass.
    :param temp: Temperature (K).
    :param k_b: Boltzmann constant (cal/K).
    :param mass: Mass (kg).
    :return: Random value distributed according to Maxwell-Boltzmann distribution.
    """
    var = (k_b * 4.184 * temp) / mass
    velocity = np.random.multivariate_normal(np.zeros(3), np.diag([var]*3))
    return velocity


def rdkit_hmc(args: Args) -> None:
    """
    Metropolis-Hastings conformational search using RDKit.
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

    # Generate initial conformation
    # Here, we consider the variables of interest, q, to effectively be the atomic coordinates
    current_q = copy.deepcopy(mol)
    AllChem.EmbedMolecule(current_q, maxAttempts=args.max_attempts)
    num_atoms = current_q.GetNumAtoms()

    # Mass in kg
    mass = np.array([mol.GetAtomWithIdx(i).GetMass()/(1000.*avogadro) for i in range(num_atoms)])

    # Add the first conformation to the list
    conformation_molecules = [current_q]

    # ###### Testing ###############
    # test = copy.deepcopy(mol)
    # AllChem.EmbedMultipleConfs(test, numConfs=int(args.L/100))
    # ######################################

    print(f'Running HMC steps...')
    num_accepted = 0
    count = 0
    for _ in tqdm(range(args.num_steps)):
        # Set the current position variables
        q = copy.deepcopy(current_q)

        # Generate random momentum values by sampling velocities from the Maxwell-Boltzmann distribution
        # Momentum is in kg * m / s
        # Set the velocity center of mass to zero
        v = np.array([maxwell_boltzmann(args.temp, k_b, mass[i]) for i in range(num_atoms)])
        p = np.array([v[i]*mass[i] for i in range(num_atoms)])
        v_cm = np.sum(p, axis=0) / sum(mass)
        for i in range(num_atoms):
            v[i] -= v_cm
        p = np.array([v[i] * mass[i] for i in range(num_atoms)])
        current_p = p

        # Make a half-step for momentum at the beginning
        # Note: the gradient is in kcal/mol/Angstrom, so we convert it to Newtons: 1 kg * m / s^2
        _, grad_u = calc_energy_grad(q)
        grad_u *= (1000.0 * 4.184 * 10e10 / avogadro)
        p = p - args.epsilon * grad_u/2.

        # Alternate full steps for position and momentum
        for i in range(args.L):
            # Make a full step for the position
            c = q.GetConformer()
            pos = c.GetPositions()
            v = np.array([p[i] / mass[i] for i in range(num_atoms)])
            v *= 10e10  # Convert to Angstroms / s
            pos = pos + args.epsilon * v

            # Save updated atomic coordinates to the conformation object
            for j in range(len(pos)):
                c.SetAtomPosition(j, Point3D(pos[j][0], pos[j][1], pos[j][2]))

            # if i % 100 == 0:
            #     # Testing ###################
            #     c = test.GetConformers()[count]
            #     for j in range(len(pos)):
            #         c.SetAtomPosition(j, Point3D(pos[j][0], pos[j][1], pos[j][2]))
            #     count += 1
            #     #############################

            # Make a full step for the momentum, except at the end of the trajectory
            if i != args.L - 1:
                energy, grad_u = calc_energy_grad(q)
                grad_u *= (1000.0 * 4.184 * 10e10 / avogadro)
                p = p - args.epsilon * grad_u

        # Make a half step for momentum at the end
        _, grad_u = calc_energy_grad(q)
        grad_u *= (1000.0 * 4.184 * 10e10 / avogadro)
        p = p - args.epsilon * grad_u/2.

        # print(rdmolfiles.MolToPDBBlock(test), file=open("traj.pdb", "w+"))

        # Negate the momentum at the end of the trajectory to make the proposal symmetric
        p *= -1.0

        # Evaluate potential and kinetic energies at start and end of the trajectory
        # Energies are in Joules
        current_u, _ = calc_energy_grad(current_q)
        current_u *= (1000.0 * 4.184 / avogadro)
        current_k = sum([((np.linalg.norm(current_p[i]))**2 / (2. * mass[i])) for i in
                         range(num_atoms)])
        proposed_u, _ = calc_energy_grad(q)
        proposed_u *= (1000.0 * 4.184 / avogadro)
        proposed_k = sum([((np.linalg.norm(p[i]))**2 / (2. * mass[i])) for i in
                          range(num_atoms)])

        prob_ratio = math.exp((current_u - proposed_u + current_k - proposed_k) / (k_b*args.temp*4.184))
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
