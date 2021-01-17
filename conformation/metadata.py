""" Generate metadata. """
import itertools
import json
import pickle
import os
import re
from typing import List, Union

import numpy as np
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdmolops, rdPartialCharges, rdForceFieldHelpers, rdchem
from scipy import sparse
# noinspection PyPackageRequirements
from tap import Tap
import torch

from conformation.graph_data import Data
from conformation.relational import RelationalNetwork
from conformation.train_args_relational import Args as graph_Args
from conformation.relational_utils import load_relational_checkpoint


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to directory containing data
    save_dir: str  # Path to directory containing output file
    mpnn: bool = False  # Whether or not to produce metadata for graph neural network training
    autoencoder: bool = False  # Whether or not to produce metadata for graph neural network autoencoder training
    cnf: bool = False  # Whether or not to produce metadata for conditional normalizing flow training
    gnf: bool = False  # Whether or not to produce metadata for graph normalizing flow training
    gmm: bool = False  # Whether or not to produce metadata for gmm toy example
    graph_model_path: str = None  # Path to saved graph model (cnf = True)
    smiles_dir: str = None  # Path to directory containing smiles strings (mpnn or cnf = True)
    binary_dir: str = None  # Path to directory containing RDKit mol binary files (mpnn = True)
    atom_types: List[int] = [1, 6, 7, 8, 9]  # Graph neural net allowed atom types (cnf = True)
    bond_types: List[float] = [0., 1., 1.5, 2., 3.]  # Graph neural net allowed bond types (cnf = True)


def to_one_hot(x: int, vals: Union[List, range]) -> List:
    """
    Return a one-hot vector.
    :param x: Data integer.
    :param vals: List of possible data values.
    :return: One-hot vector as list.
    """
    return [x == v for v in vals]


def metadata(args: Args) -> None:
    """
    Create metadata folder and file.
    :param args: Folder name info.
    :return None.
    """
    os.makedirs(args.save_dir)
    if args.cnf:
        # Load trained relational network
        state = torch.load(args.graph_model_path, map_location=lambda storage, loc: storage)
        loaded_args = graph_Args().from_dict(state['args'])
        loaded_state_dict = state['state_dict']

        graph_model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                                        loaded_args.num_vertex_features, loaded_args.final_linear_size,
                                        loaded_args.final_output_size, cnf=True)
        graph_model.load_state_dict(loaded_state_dict)

        if torch.cuda.is_available():
            graph_model = graph_model.cuda()

        graph_model.eval()

        os.makedirs(os.path.join(args.save_dir, "conditions"))

    if args.gnf:
        graph_model, loaded_args = load_relational_checkpoint(args.graph_model_path, graph_Args())
        graph_model.gnf_metadata = True
        if torch.cuda.is_available():
            graph_model = graph_model.cuda()
        graph_model.eval()
        os.makedirs(os.path.join(args.save_dir, "targets"))

    data = []
    conditional_dict = dict()
    uid_dict = dict()
    binary_dict = dict()
    uid = 0
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            path = os.path.join(root, f)
            if args.mpnn:
                if args.autoencoder:
                    dashes = [m.start() for m in re.finditer("-", f)]
                    molecule_name = f[dashes[1] + 1:dashes[2]]
                else:
                    molecule_name = f[[m.start() for m in re.finditer("-", f)][1] + 1:f.find(".")]
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                binary_path = os.path.join(args.binary_dir, molecule_name + ".bin")
                data.append({'smiles': smiles, 'target': path, 'uid': uid, 'binary': binary_path})
                uid_dict[uid] = smiles
                binary_dict[uid] = binary_path
                uid += 1
            elif args.gnf:
                dashes = [m.start() for m in re.finditer("-", f)]
                molecule_name = f[dashes[1] + 1:dashes[2]]
                binary_path = os.path.join(args.binary_dir, molecule_name + ".bin")
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]

                sample = Data()
                # Molecule from binary
                # noinspection PyUnresolvedReferences
                mol = Chem.Mol(open(binary_path, "rb").read())
                num_atoms = mol.GetNumAtoms()

                # Target
                # noinspection PyUnboundLocalVariable
                # Target: 1-D tensor representing average inter-atomic distance for each edge
                target = np.load(path)
                sample.y = torch.tensor(target, dtype=torch.float)

                # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
                complete_graph = np.ones([num_atoms, num_atoms])  # Create an auxiliary complete graph
                complete_graph = np.triu(complete_graph,
                                         k=1)  # Compute an upper triangular matrix of the complete graph
                complete_graph = sparse.csc_matrix(complete_graph)  # Compute a csc style sparse matrix from this graph
                row, col = complete_graph.nonzero()  # Extract row and column indices corresponding to non-zero entries
                row = torch.tensor(row, dtype=torch.long)
                col = torch.tensor(col, dtype=torch.long)
                sample.edge_index = torch.stack([row, col])  # Edge connectivity in COO format (all possible edges)

                # Edge features
                edge_features = []

                edge_count = 0
                for a, b in itertools.combinations(list(np.arange(num_atoms)), 2):
                    bond_feature = []
                    bond = mol.GetBondBetweenAtoms(int(a), int(b))
                    if bond is None:
                        # noinspection PyUnboundLocalVariable
                        if loaded_args.bond_type:
                            bond_feature += [1] + [0] * len(loaded_args.bond_types)

                        if loaded_args.conjugated:
                            bond_feature += [0]

                        if loaded_args.bond_ring:
                            bond_feature += [0]

                        if loaded_args.bond_stereo:
                            bond_feature += [0] * len(loaded_args.bond_stereo_types)

                        if loaded_args.shortest_path:
                            path_len = len(rdmolops.GetShortestPath(mol, int(a), int(b))) - 1
                            bond_feature += to_one_hot(path_len - 1, range(loaded_args.max_shortest_path_length))

                        if loaded_args.same_ring:
                            ring_info = list(mol.GetRingInfo().AtomRings())
                            membership = [int(a) in r and int(b) in r for r in ring_info]
                            if sum(membership) > 0:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if loaded_args.autoencoder:
                            # noinspection PyUnboundLocalVariable
                            bond_feature += [target[:, 0][edge_count]]

                    else:
                        if loaded_args.bond_type:
                            bond_feature += [0]
                            bond_feature += to_one_hot(bond.GetBondTypeAsDouble(), loaded_args.bond_types)

                        if loaded_args.conjugated:
                            bond_feature += [bond.GetIsConjugated()]

                        if loaded_args.bond_ring:
                            bond_feature += [bond.IsInRing()]

                        if loaded_args.bond_stereo:
                            bond_feature += to_one_hot(bond.GetStereo(), loaded_args.bond_stereo_types)

                        if loaded_args.shortest_path:
                            path_len = len(rdmolops.GetShortestPath(mol, int(a), int(b))) - 1
                            bond_feature += to_one_hot(path_len - 1, range(loaded_args.max_shortest_path_length))

                        if loaded_args.same_ring:
                            ring_info = list(mol.GetRingInfo().AtomRings())
                            membership = [int(a) in r and int(b) in r for r in ring_info]
                            if sum(membership) > 0:
                                bond_feature += [1]
                            else:
                                bond_feature += [0]

                        if loaded_args.autoencoder:
                            bond_feature += [target[:, 0][edge_count]]

                    edge_count += 1

                    edge_features.append(bond_feature)

                sample.edge_attr = torch.tensor(edge_features, dtype=torch.float)

                # Vertex features
                # List to hold all vertex features
                vertex_features = []

                pt = Chem.GetPeriodicTable()

                if loaded_args.partial_charge:
                    rdPartialCharges.ComputeGasteigerCharges(mol)

                mmff_p = None
                if loaded_args.mmff_atom_types_one_hot:
                    # AllChem.EmbedMolecule(mol, maxAttempts=100000)
                    # AllChem.MMFFOptimizeMolecule(mol)
                    mmff_p = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)

                if loaded_args.assign_stereo:
                    rdmolops.AssignStereochemistryFrom3D(mol)

                for i in range(num_atoms):
                    atom = mol.GetAtomWithIdx(i)
                    atom_feature = []

                    if loaded_args.atomic_num:
                        atom_feature += to_one_hot(atom.GetAtomicNum(), loaded_args.atom_types)

                    if loaded_args.valence:
                        atom_feature += to_one_hot(atom.GetTotalValence(), loaded_args.valence_types)

                    if loaded_args.aromatic:
                        atom_feature += [atom.GetIsAromatic()]

                    if loaded_args.hybridization:
                        atom_feature += to_one_hot(atom.GetHybridization(), loaded_args.hybridization_types)

                    if loaded_args.partial_charge:
                        gc = float(atom.GetProp('_GasteigerCharge'))
                        if not np.isfinite(gc):
                            gc = 0.0
                        atom_feature += [gc]

                    if loaded_args.formal_charge:
                        atom_feature += to_one_hot(atom.GetFormalCharge(), loaded_args.charge_types)

                    if loaded_args.r_covalent:
                        atom_feature += [pt.GetRcovalent(atom.GetAtomicNum())]

                    if loaded_args.r_vanderwals:
                        atom_feature += [pt.GetRvdw(atom.GetAtomicNum())]

                    if loaded_args.default_valence:
                        atom_feature += to_one_hot(pt.GetDefaultValence(atom.GetAtomicNum()), loaded_args.valence_types)

                    if loaded_args.rings:
                        atom_feature += [atom.IsInRingSize(r) for r in range(3, loaded_args.max_ring_size + 1)]

                    if loaded_args.chirality:
                        atom_feature += to_one_hot(atom.GetChiralTag(), loaded_args.chi_types)

                    if loaded_args.mmff_atom_types_one_hot:
                        if mmff_p is None:
                            atom_feature += [0] * len(loaded_args.mmff94_atom_types)
                        else:
                            atom_feature += to_one_hot(mmff_p.GetMMFFAtomType(i), loaded_args.mmff94_atom_types)

                    if loaded_args.degree:
                        atom_feature += to_one_hot(atom.GetDegree(), loaded_args.degree_types)

                    if loaded_args.num_hydrogen:
                        atom_feature += to_one_hot(atom.GetTotalNumHs(), loaded_args.num_hydrogen_types)

                    if loaded_args.num_radical_electron:
                        atom_feature += to_one_hot(atom.GetNumRadicalElectrons(),
                                                   loaded_args.num_radical_electron_types)

                    vertex_features.append(atom_feature)

                sample.x = torch.tensor(vertex_features, dtype=torch.float)

                # Pass through graph model to get embeddings
                with torch.no_grad():
                    if torch.cuda.is_available():
                        sample.x = sample.x.cuda()
                        sample.edge_attr = sample.edge_attr.cuda()
                    # noinspection PyUnboundLocalVariable
                    v_i_in, e_ij_in = graph_model(sample)
                    v_i_in = v_i_in.cpu().numpy()
                    e_ij_in = e_ij_in.cpu().numpy()
                node_target_path = os.path.join(args.save_dir, "targets", molecule_name + "-node-embedding")
                edge_target_path = os.path.join(args.save_dir, "targets", molecule_name + "-edge-embedding")
                np.save(node_target_path, v_i_in)
                np.save(edge_target_path, e_ij_in)
                data.append({'smiles': smiles, 'target': [node_target_path, edge_target_path],
                             'uid': uid, 'binary': binary_path})
                uid_dict[uid] = smiles
                binary_dict[uid] = binary_path
                uid += 1
                print(f'# Conformations Processed: {uid}')

            elif args.cnf:
                molecule_name = f[[m.end() for m in re.finditer("-", f)][-1]:f.find(".")]
                with open(os.path.join(args.smiles_dir, molecule_name + ".smiles")) as tmp:
                    smiles = tmp.readlines()[0].split()[0]
                if molecule_name not in conditional_dict:
                    sample = Data()  # Create data object

                    # Molecule from SMILES string
                    mol = Chem.MolFromSmiles(smiles)
                    mol = Chem.AddHs(mol)
                    num_atoms = mol.GetNumAtoms()

                    # Compute edge connectivity in COO format corresponding to a complete graph on num_nodes
                    complete_graph = np.ones([num_atoms, num_atoms])  # Create an auxiliary complete graph
                    complete_graph = np.triu(complete_graph,
                                             k=1)  # Compute an upper triangular matrix of the complete graph
                    complete_graph = sparse.csc_matrix(
                        complete_graph)  # Compute a csc style sparse matrix from this graph
                    row, col = complete_graph.nonzero()  # Extract row and column indices of non-zero entries
                    row = torch.tensor(row, dtype=torch.long)
                    col = torch.tensor(col, dtype=torch.long)
                    sample.edge_index = torch.stack([row, col])  # Edge connectivity in COO format (all possible edges)

                    # Edge features
                    # Create one-hot encoding
                    one_hot_bond_features = np.zeros((len(args.bond_types), len(args.bond_types)))
                    np.fill_diagonal(one_hot_bond_features, 1.)
                    bond_to_one_hot = dict()
                    for i in range(len(args.bond_types)):
                        bond_to_one_hot[args.bond_types[i]] = one_hot_bond_features[i]

                    # Extract atom indices participating in bonds and bond types
                    bonds = []
                    bond_types = []
                    for bond in mol.GetBonds():
                        bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                        bond_types.append([bond_to_one_hot[bond.GetBondTypeAsDouble()]])

                    # Compute edge attributes: 1 indicates presence of bond, 0 no bond.
                    full_edges = [list(sample.edge_index[:, i].numpy()) for i in range(sample.edge_index.shape[1])]
                    no_bond = np.concatenate([np.array([0]), bond_to_one_hot[0]])
                    a = np.array([1])
                    edge_attr = [
                        np.concatenate([a, bond_types[bonds.index(full_edges[i])][0]]) if full_edges[i] in bonds else
                        no_bond for i in range(len(full_edges))]
                    sample.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                    # TODO: add additional shortest path length bond features

                    # Vertex features: one-hot representation of atomic number
                    # Create one-hot encoding
                    one_hot_vertex_features = np.zeros((len(args.atom_types), len(args.atom_types)))
                    np.fill_diagonal(one_hot_vertex_features, 1.)
                    atom_to_one_hot = dict()
                    for i in range(len(args.atom_types)):
                        atom_to_one_hot[args.atom_types[i]] = one_hot_vertex_features[i]

                    one_hot_features = np.array([atom_to_one_hot[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
                    sample.x = torch.tensor(one_hot_features, dtype=torch.float)

                    with torch.no_grad():
                        if torch.cuda.is_available():
                            sample.x = sample.x.cuda()
                            sample.edge_attr = sample.edge_attr.cuda()
                        # noinspection PyUnboundLocalVariable
                        embedding = graph_model(sample).cpu().numpy()
                    np.save(os.path.join(args.save_dir, "conditions", molecule_name), embedding)

                    conditional_dict[molecule_name] = os.path.join(args.save_dir, "conditions", molecule_name + ".npy")

                data.append({'smiles': smiles, 'path': path, 'condition': conditional_dict[molecule_name]})
            elif args.gmm:
                if f[:5] == "gmm_s":
                    path = os.path.join(root, f)
                    condition_path = os.path.join(root, "gmm_conditions_" + f[12:-4] + ".ptx")
                    data.append({'path': path, 'condition': condition_path})
            else:
                data.append({'path': path})
    if len(uid_dict) > 0:
        pickle.dump(uid_dict, open(os.path.join(args.save_dir, "uid_dict.p"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if len(binary_dict) > 0:
        pickle.dump(binary_dict, open(os.path.join(args.save_dir, "binary_dict.p"), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(data, open(os.path.join(args.save_dir, os.path.basename(args.save_dir) + ".json"), "w"), indent=4,
              sort_keys=True)
