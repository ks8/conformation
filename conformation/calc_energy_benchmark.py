""" calcEnergy and calcGrad benchmarking. """
import numpy as np
import random
import time
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdForceFieldHelpers
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
    num_steps: int = 1000  # Number of HMC steps
    num_repetitions: int = 5  # Number of repetitions of timing
    perturbation: float = 0.0001  # Perturbation magnitude


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


# noinspection PyUnresolvedReferences
def calc_energy_grad_new(mol: rdchem.Mol, force_field) -> Tuple[float, np.ndarray]:
    """
    New compute MMFF energy and gradient.
    :param mol: Molecule.
    :param force_field: Force field object.
    :return: stuff.
    """
    pos = tuple(mol.GetConformer().GetPositions().flatten())

    energy = force_field.CalcEnergy(pos)
    grad = force_field.CalcGrad(pos)
    grad = np.reshape(np.array(grad), [int(len(grad)/3), 3])

    return energy, grad


# noinspection PyUnresolvedReferences
def perturbation(mol: rdchem.Mol, perturbation_magnitude) -> rdchem.Mol:
    """
    Small coords perturbations.
    :param mol: Molecule.
    :param perturbation_magnitude: Perturbation magnitude.
    :return: Perturbed molecule.
    """
    c = mol.GetConformer()
    pos = mol.GetConformer().GetPositions()
    for j in range(len(pos)):
        c.SetAtomPosition(j, Point3D(pos[j][0] + random.uniform(-perturbation_magnitude, perturbation_magnitude),
                                     pos[j][1] + random.uniform(-perturbation_magnitude, perturbation_magnitude),
                                     pos[j][2] + random.uniform(-perturbation_magnitude, perturbation_magnitude)))

    return mol


def calc_energy_benchmark(args: Args) -> None:
    """
    Calc energy benchmark.
    :param args: System args.
    :return: None
    """
    # noinspection PyUnresolvedReferences
    mol = Chem.Mol(open(args.bin_path, "rb").read())
    mol.RemoveAllConformers()
    AllChem.EmbedMolecule(mol)

    times = []
    for _ in tqdm(range(args.num_repetitions)):
        start_time = time.time()
        for _ in tqdm(range(args.num_steps)):
            mol = perturbation(mol, args.perturbation)
            calc_energy_grad(mol)
        end_time = time.time()
        times.append(end_time - start_time)
    times = np.array(times)
    print(f'Old method avg (s): {np.mean(times)} +/- {np.std(times)}')

    times = []
    for _ in tqdm(range(args.num_repetitions)):
        start_time = time.time()
        mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
        mmff_f = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mmff_p)
        for _ in tqdm(range(args.num_steps)):
            mol = perturbation(mol, args.perturbation)
            calc_energy_grad_new(mol, mmff_f)
        end_time = time.time()
        times.append(end_time - start_time)
    times = np.array(times)
    print(f'Old method avg (s): {np.mean(times)} +/- {np.std(times)}')
