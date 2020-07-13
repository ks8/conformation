""" Generate samples from trained normalizing flow. """
import itertools
import numpy as np
import os
from typing import List

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from conformation.flows import NormalizingFlowModel


def sample(model: NormalizingFlowModel, smiles: str, save_dir: str, num_atoms: int, offset: float,
           num_layers: int, num_samples: int, dihedral: bool, dihedral_vals: List[int]) -> None:
    """
    Generate samples from trained normalizing flow.
    :param dihedral_vals:
    :param dihedral:
    :param model: PyTorch model.
    :param smiles: Molecular SMILES string.
    :param save_dir: Directory for saving generated conformations.
    :param num_atoms: Total number of atoms in the molecule.
    :param offset: Distance bounds matrix offset.
    :param num_samples: Number of conformations to generate.
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
        tmp = Chem.MolFromSmiles(smiles)
        tmp = Chem.AddHs(tmp)

        for j in range(num_samples):
            gen_sample = model.sample(num_layers)
            distmat = np.zeros([num_atoms, num_atoms])
            boundsmat = np.zeros([num_atoms, num_atoms])
            indices = []
            for m, n in itertools.combinations(np.arange(num_atoms), 2):
                indices.append((m, n))
            for i in range(len(gen_sample)):
                distmat[indices[i][0], indices[i][1]] = gen_sample[i].item()
                distmat[indices[i][1], indices[i][0]] = distmat[indices[i][0], indices[i][1]]

                boundsmat[indices[i][0], indices[i][1]] = distmat[indices[i][0], indices[i][1]] + offset
                boundsmat[indices[i][1], indices[i][0]] = distmat[indices[i][1], indices[i][0]] - offset

            # Set the bounds matrix
            ps.SetBoundsMat(boundsmat)

            # Create a conformation from the distance bounds matrix
            # noinspection PyUnusedLocal
            AllChem.EmbedMolecule(tmp, params=ps)

            try:
                # Test that the conformation is valid
                c = tmp.GetConformer()

                # Try saving the molecule and reloading
                print(Chem.rdmolfiles.MolToPDBBlock(tmp), file=open(os.path.join(save_dir, "test.pdb"), "w+"))
                test_mol = AllChem.MolFromPDBFile(os.path.join(save_dir, "test.pdb"), removeHs=False)
                test_mol.GetConformer()

                # Set the conformer Id and increment the conformation counter
                c.SetId(counter)
                counter += 1

                # Add the conformer to the overall molecule object
                mol.AddConformer(c)

                np.savetxt(os.path.join(save_dir, "distmat", "distmat-" + str(counter) + ".txt"), distmat)  # TODO: move inside try block and have while loop generating correct number of valid configs

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
            except AttributeError:
                continue

        # Print the conformations to a PDB file #TODO: Issue that some conformations.pdb files give valence errors....why does deleting the first conformation work?
        print(Chem.rdmolfiles.MolToPDBBlock(mol), file=open(os.path.join(save_dir, "conformations.pdb"), "w+"))
