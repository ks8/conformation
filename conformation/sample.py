""" Generate samples from trained normalizing flow. """
import numpy as np
import os
from typing import List

# noinspection PyUnresolvedReferences
import rdkit
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdDistGeom
import torch

from conformation.flows import NormalizingFlowModel


def sample(model: NormalizingFlowModel, smiles: str, save_dir: str, num_atoms: int, offset: float,
           num_layers: int, num_test_samples: int, dihedral: bool, dihedral_vals: List[int]) -> None:
    """
    Generate samples from trained normalizing flow.
    :param dihedral_vals:
    :param dihedral:
    :param model: PyTorch model.
    :param smiles: Molecular SMILES string.
    :param save_dir: Directory for saving generated conformations.
    :param num_atoms: Total number of atoms in the molecule.
    :param offset: Distance bounds matrix offset.
    :param num_test_samples: Number of conformations to generate.
    :param num_layers: Number of layers to use in generating sample.
    :return: None.
    """
    os.makedirs(os.path.join(save_dir, "distmat"))
    os.makedirs(os.path.join(save_dir, "properties"))

    # Conformation counter
    counter = 0

    with torch.no_grad():
        model.eval()
        num_atoms = num_atoms

        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDG()

        # Create a random conformation object
        # noinspection PyUnusedLocal
        tmp = Chem.MolFromSmiles(smiles)
        tmp = Chem.AddHs(mol)

        for j in range(num_test_samples):
            gen_sample = model.sample(num_layers)
            distmat = np.zeros([num_atoms, num_atoms])
            boundsmat = np.zeros([num_atoms, num_atoms])
            indices = []
            for m in range(num_atoms):
                for n in range(1, num_atoms):
                    if n > m:
                        indices.append((m, n))
            for i in range(len(gen_sample)):
                distmat[indices[i][0], indices[i][1]] = gen_sample[i].item()
                distmat[indices[i][1], indices[i][0]] = distmat[indices[i][0], indices[i][1]]

                boundsmat[indices[i][0], indices[i][1]] = distmat[indices[i][0], indices[i][1]] + offset
                boundsmat[indices[i][1], indices[i][0]] = distmat[indices[i][1], indices[i][0]] - offset
            np.savetxt(os.path.join(save_dir, "distmat", "distmat-" + str(j) + ".txt"), distmat)

            # Set the bounds matrix
            ps.SetBoundsMat(boundsmat)

            # Create a conformation from the distance bounds matrix
            # noinspection PyUnusedLocal
            AllChem.EmbedMolecule(tmp, params=ps)

            try:
                # Test that the conformation is valid
                c = tmp.GetConformer()

                # Set the conformer Id and increment the conformation counter
                c.SetId(counter)
                counter += 1

                # Add the conformer to the overall molecule object
                mol.AddConformer(c)

                # Compute properties of the conformation
                res = AllChem.MMFFOptimizeMoleculeConfs(tmp, maxIters=0)
                with open(os.path.join(save_dir, "properties", "energy-rms-dihedral-" + str(counter) + ".txt"), "w") \
                        as o:
                    o.write("energy: " + str(res[0][1]))
                    o.write('\n')
                    o.write("rms: " + "nan")
                    o.write('\n')
                    if dihedral:
                        dihedral_val = Chem.rdMolTransforms.GetDihedralRad(c, dihedral_vals[0], dihedral_vals[1],
                                                                           dihedral_vals[2], dihedral_vals[3])
                    else:
                        dihedral_val = "nan"
                    o.write("dihedral: " + str(dihedral_val))

            except ValueError:
                continue

        # Print the conformations to a PDB file
        print(Chem.rdmolfiles.MolToPDBBlock(mol), file=open(os.path.join(save_dir, "conformations.pdb"), "w+"))
