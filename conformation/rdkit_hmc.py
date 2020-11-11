""" Hamiltonian Monte Carlo conformational search using RDKit. """
import copy
from logging import Logger
import math
import numpy as np
import os
import random
import time
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdForceFieldHelpers
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.ForceField import rdForceField
# noinspection PyUnresolvedReferences
from rdkit.Geometry.rdGeometry import Point3D
# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    bin_path: str  # Path to RDKit binary file containing molecule
    max_attempts: int = 10000  # Max number of embedding attempts
    num_minimization_iters: int = 0  # Number of minimization steps
    temp: float = 300.0  # Temperature for computing Boltzmann probabilities
    epsilon: float = 1  # Leapfrog step size in femtoseconds
    L: int = 10  # Number of leapfrog steps
    num_steps: int = 1  # Number of HMC steps
    subsample_frequency: int = 1  # Frequency at which configurations are saved from MH steps
    log_frequency: int = 100  # Log frequency
    save_dir: str  # Path to directory containing output files


# noinspection PyUnresolvedReferences
def calc_energy_grad(pos: np.ndarray, force_field: rdForceField.ForceField) -> Tuple[float, np.ndarray]:
    """
    Compute MMFF energy and gradient.
    :param pos: Atomic coordinates.
    :param force_field: RDKit force field.
    :return: MMFF energy and gradient, where the energy is kcal/mol and the gradient is kcal/mol/Angstrom.
    """
    pos = tuple(pos.flatten())

    energy = force_field.CalcEnergy(pos)
    grad = force_field.CalcGrad(pos)
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


# noinspection PyUnresolvedReferences,PyPep8Naming
def hmc_step(current_q: rdchem.Mol, force_field: rdForceField.ForceField, temp: float, k_b: float,
             avogadro: float, mass: np.ndarray, num_atoms: int, epsilon: float, L: int) -> \
        Tuple[bool, rdchem.Mol, float]:
    """
    Run a single Hamiltonian Monte Carlo step.
    :param current_q: Current configuration as specified in an RDKit molecule object.
    :param force_field: RDKit molecule object force field.
    :param temp: Temperature.
    :param k_b: Boltzmann's constant.
    :param avogadro: Avogadro's number.
    :param mass: Numpy array containing the mass of each atom.
    :param num_atoms: Number of atoms.
    :param epsilon: Leapfrog step size.
    :param L: Number of leapfrog steps.
    :return: Whether or not trial move is accepted, the updated configuration, and the updated energy.
    """
    # Set the position variables
    current_pos = current_q.GetConformer().GetPositions()
    q = copy.deepcopy(current_q)
    pos = q.GetConformer().GetPositions()

    # Generate random momentum values by sampling velocities from the Maxwell-Boltzmann distribution
    # Momentum is in kg * m / s
    # Set the velocity center of mass to zero
    v = np.array([maxwell_boltzmann(temp, k_b, mass[i]) for i in range(num_atoms)])
    p = np.array([v[i] * mass[i] for i in range(num_atoms)])
    v_cm = np.sum(p, axis=0) / sum(mass)
    for i in range(num_atoms):
        v[i] -= v_cm
    p = np.array([v[i] * mass[i] for i in range(num_atoms)])
    current_p = p

    # Make a half-step for momentum at the beginning
    # Note: the gradient is in kcal/mol/Angstrom, so we convert it to Newtons: 1 kg * m / s^2
    _, grad_u = calc_energy_grad(pos, force_field)
    grad_u *= (1000.0 * 4.184 * 1e10 / avogadro)
    p = p - epsilon * grad_u / 2.

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        c = q.GetConformer()
        pos = c.GetPositions()
        v = np.array([p[i] / mass[i] for i in range(num_atoms)])
        v *= 1e10  # Convert to Angstroms / s
        pos = pos + epsilon * v

        # Save updated atomic coordinates to the conformation object
        for j in range(len(pos)):
            c.SetAtomPosition(j, Point3D(pos[j][0], pos[j][1], pos[j][2]))

        # Make a full step for the momentum, except at the end of the trajectory
        if i != L - 1:
            energy, grad_u = calc_energy_grad(pos, force_field)
            grad_u *= (1000.0 * 4.184 * 1e10 / avogadro)
            p = p - epsilon * grad_u

    # Make a half step for momentum at the end
    _, grad_u = calc_energy_grad(pos, force_field)
    grad_u *= (1000.0 * 4.184 * 1e10 / avogadro)
    p = p - epsilon * grad_u / 2.

    # Negate the momentum at the end of the trajectory to make the proposal symmetric
    p *= -1.0

    # Evaluate potential and kinetic energies at start and end of the trajectory
    # Energies are in Joules
    current_u, _ = calc_energy_grad(current_pos, force_field)
    current_u *= (1000.0 * 4.184 / avogadro)
    current_k = sum([((np.linalg.norm(current_p[i])) ** 2 / (2. * mass[i])) for i in
                     range(num_atoms)])
    proposed_u, _ = calc_energy_grad(pos, force_field)
    proposed_u *= (1000.0 * 4.184 / avogadro)
    proposed_k = sum([((np.linalg.norm(p[i])) ** 2 / (2. * mass[i])) for i in
                      range(num_atoms)])

    # Current energy in kcal/mol
    current_energy = current_u / (1000.0 * 4.184 / avogadro)

    prob_ratio = math.exp((current_u - proposed_u + current_k - proposed_k) / (k_b * temp * 4.184))
    mu = random.uniform(0, 1)
    accepted = False
    if mu <= prob_ratio:
        current_q = q
        current_energy = proposed_u / (1000.0 * 4.184 / avogadro)
        accepted = True

    return accepted, current_q, current_energy


def rdkit_hmc(args: Args, logger: Logger) -> None:
    """
    Metropolis-Hastings conformational search using RDKit.
    :param args: System arguments.
    :param logger: System logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    # Define constants
    k_b = 3.297e-24  # Boltzmann constant in cal/K
    avogadro = 6.022e23
    args.epsilon *= 1e-15

    # Load molecule
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.bin_path, "rb").read())
    mol.RemoveAllConformers()

    debug(f'Starting search...')

    # Generate initial conformation
    # Here, we consider the variables of interest, q, to effectively be the atomic coordinates
    current_q = copy.deepcopy(mol)
    AllChem.EmbedMolecule(current_q, maxAttempts=args.max_attempts)
    AllChem.MMFFOptimizeMoleculeConfs(current_q, maxIters=args.num_minimization_iters)
    num_atoms = current_q.GetNumAtoms()

    # Compute force field information
    mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(current_q)
    force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(current_q, mmff_p)

    # Masses in kg
    mass = np.array([mol.GetAtomWithIdx(i).GetMass()/(1000.*avogadro) for i in range(num_atoms)])

    # Add the first conformation to the list
    energy, _ = calc_energy_grad(current_q, force_field)
    conformation_molecules = [current_q]
    energies = [energy]
    all_conformation_molecules = [current_q]
    all_energies = [energy]

    debug(f'Running HMC steps...')
    start_time = time.time()
    num_accepted = 0
    for step in tqdm(range(args.num_steps)):
        accepted, current_q, current_energy = hmc_step(current_q, force_field, args.temp, k_b, avogadro, mass,
                                                       num_atoms, args.epsilon, args.L)
        if accepted:
            conformation_molecules.append(current_q)
            energies.append(current_energy)
            num_accepted += 1

        if step % args.subsample_frequency == 0:
            all_conformation_molecules.append(current_q)
            all_energies.append(current_energy)

        if step % args.log_frequency == 0:
            debug(f'Number of conformations identified: {len(conformation_molecules)}')
            debug(f'% Moves accepted: {float(num_accepted) / float(step + 1) * 100.0}')
    end_time = time.time()
    debug(f'Total Time(s): {end_time - start_time}')
    debug(f'Number of conformations identified: {len(conformation_molecules)}')
    debug(f'% Moves accepted: {float(num_accepted) / float(args.num_steps) * 100.0}')

    # Discover the rotatable bonds
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    debug(f'Num rotatable bonds: {len(rotatable_bonds)}')

    # Save unique conformations in molecule object
    debug(f'Saving conformations...')
    for i in range(len(conformation_molecules)):
        c = conformation_molecules[i].GetConformer()
        c.SetId(i)
        mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = mol.ToBinary()
    with open(os.path.join(args.save_dir, "accepted-conformations.bin"), "wb") as b:
        b.write(bin_str)

    # Save all conformations in molecule object
    # noinspection PyUnresolvedReferences
    all_mol = Chem.Mol(open(args.bin_path, "rb").read())
    all_mol.RemoveAllConformers()
    for i in range(len(all_conformation_molecules)):
        c = all_conformation_molecules[i].GetConformer()
        c.SetId(i)
        all_mol.AddConformer(c)

    # Save molecule to binary file
    bin_str = all_mol.ToBinary()
    with open(os.path.join(args.save_dir, "all-conformations.bin"), "wb") as b:
        b.write(bin_str)
