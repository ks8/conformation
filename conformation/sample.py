""" Generate samples from trained normalizing flow. """
import itertools
import numpy as np
import os

# noinspection PyUnresolvedReferences
import rdkit
# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdDistGeom
import torch

from conformation.flows import NormalizingFlowModel


def sample(model: NormalizingFlowModel, smiles: str, save_dir: str, num_atoms: int, offset: float,
           num_layers: int, num_test_samples: int) -> None:
    """
    Generate samples from trained normalizing flow.
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
    os.makedirs(os.path.join(save_dir, "conf"))
    os.makedirs(os.path.join(save_dir, "properties"))

    distance_matrices = []

    with torch.no_grad():
        model.eval()
        num_atoms = num_atoms

        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDG()

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
            distance_matrices.append(gen_sample.cpu().numpy())

            ps.SetBoundsMat(boundsmat)

            # Generate and print conformation as PDB file
            AllChem.EmbedMolecule(mol, params=ps)

            try:
                # Compute the specified dihedral angle
                c = mol.GetConformer()

                print(Chem.rdmolfiles.MolToPDBBlock(mol), file=open(os.path.join(save_dir, "conf",
                                                                                 "conf-" + str(j) + ".txt"), "w+"))

                dihedral = Chem.rdMolTransforms.GetDihedralRad(c, 2, 0, 1, 5)

                # Compute the potential energy of the conformation
                res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=0)

                # Write property information to a text file in the "properties" folder
                with open(os.path.join(save_dir, "properties", "energy-rms-dihedral-" + str(j) + ".txt"), "w") as o:
                    o.write("energy: " + str(res[0][1]))
                    o.write('\n')
                    o.write("rms: " + "nan")
                    o.write('\n')
                    o.write("dihedral: " + str(dihedral))
            except ValueError:
                continue

        distance_matrices = np.array(distance_matrices)
        with open(os.path.join(save_dir, "corrcoef.txt"), "w") as o:
            for m, n in itertools.combinations(list(np.arange(distance_matrices.shape[1])), 2):
                o.write(str(m) + " " + str(n) + " " + str(np.corrcoef(distance_matrices[:, m],
                                                                      distance_matrices[:, n])[0][1]))
                o.write('\n')
