""" Null model for predicting pairwise distances based on minimum path length between pairs of atoms. """
import itertools
import json
from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdMolTransforms, rdmolops
from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to data directory
    data_path: str  # Path to metadata file
    uid_path: str  # Path to uid file
    save_dir: str  # Directory to save information
    batch_size: int = 10  # Batch size


def null_model(args: Args, logger: Logger) -> None:
    """
    Predict atomic distances for a molecule based on minimum path length between pairs of atoms.
    :param args: System arguments.
    :param logger: Logger.
    :return: None.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    train_data = GraphDataset(train_metadata)
    val_data = GraphDataset(validation_metadata)
    test_data = GraphDataset(test_metadata)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug('train size = {:,} | val size = {:,} | test size = {:,}'.format(
        train_data_length,
        val_data_length,
        test_data_length)
    )

    # Convert to iterator
    train_data = DataLoader(train_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()

    # Uid dictionary
    uid_dict = pickle.load(open(args.uid_path, "rb"))

    # Null model 0: use the average inter-atomic distance between all pairs of atoms across all molecules
    # to predict any pairwise distance
    # Compute average inter-atomic distance across all atoms and all molecules
    debug("training null model 0")
    avg_distances = []
    for _, _, files in os.walk(args.data_dir):
        for f in files:
            distances = np.load(os.path.join(args.data_dir, f))
            avg_distances.append(np.mean(distances[:, 0]))
    avg_distances = np.array(avg_distances)
    avg_distance = avg_distances.mean()

    # Test null model 0 on the test set
    debug("testing null model 0")
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        for batch in tqdm(test_data, total=len(test_data)):
            targets = batch.y.unsqueeze(1)
            preds = torch.zeros_like(targets)
            # Use the all-atom all-molecule average distance as the prediction for each pair of atoms
            for i in range(preds.shape[0]):
                preds[i] = avg_distance
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test error avg = {:.4e}".format(loss_avg))

    # Null model 1: compute the average distance between pairs of atoms across all molecules conditioned
    # on shortest path length, and use that to predict the distance between atoms with a given shortest
    # path length.
    debug("training null model 1")
    true_distances = []
    shortest_paths = []
    path_dict = dict()
    with torch.no_grad():
        for batch in tqdm(train_data, total=len(train_data)):
            targets = batch.y[:, 0].unsqueeze(1)
            targets_aux = targets.cpu().numpy()

            # Collect the distance between each pair of atoms
            for i in range(targets_aux.shape[0]):
                true_distances.append(targets_aux[i][0])

            # Compute the shortest path between each pair of atoms
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()  # Get the uid
                smiles = uid_dict[uid]  # Access the SMILES via the uid
                mol = Chem.MolFromSmiles(smiles)  # Create a molecule object
                mol = Chem.AddHs(mol)
                # Iterate through all pairs of atoms in the molecule and compute the shortest path length
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    shortest_paths.append(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1)

    # Convert the lists of atomic distances and shortest paths to numpy arrays
    true_distances = np.array(true_distances)
    shortest_paths = np.array(shortest_paths)

    # Create a dictionary for which the keys are shortest path lengths and the values are the average
    # distance between pairs of atoms in the train set which have that shortest path length
    for i in range(shortest_paths.shape[0]):
        if shortest_paths[i] in path_dict:
            path_dict[shortest_paths[i]].append(true_distances[i])
        else:
            path_dict[shortest_paths[i]] = []
            path_dict[shortest_paths[i]].append(true_distances[i])
    for key in path_dict:
        path_dict[key] = np.array(path_dict[key]).mean()

    # Test null model 1
    debug("testing null model 1")
    path_lengths = []
    losses = []
    loss_func_aux = torch.nn.MSELoss(reduction='none')  # Use to analyze individual losses
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        for batch in tqdm(test_data, total=len(test_data)):
            targets = batch.y[:, 0].unsqueeze(1)
            preds = []
            # Loop through each molecule
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                # Predict distance based on shortest path length for each pair of atoms, keeping track
                # of both the predictions and the shortest path lengths
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    preds.append(path_dict[len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1])
                    path_lengths.append(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1)
            preds = torch.tensor(np.array(preds)).unsqueeze(1)
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
            loss_aux = loss_aux.cpu().numpy()
            for i in range(loss_aux.shape[0]):
                losses.append(loss_aux[i][0])
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test error avg = {:.4e}".format(loss_avg))

    # Plot loss as a function of shortest path length
    path_lengths = np.array(path_lengths)
    losses = np.array(losses)
    plt.plot(path_lengths, losses, 'bo', markersize=0.5)
    plt.title("Error vs Path Length")
    plt.ylabel("|True - Predicted|")
    plt.xlabel("Shortest Path")
    plt.savefig(os.path.join(args.save_dir, "null-model-1-error-vs-path"))

    # Save errors
    np.save(os.path.join(args.save_dir, "null-model-1-errors"), losses)

    # Null model 2: compute the average distance between pairs of atoms across all molecules conditioned
    # on shortest path length as well as atom types, and use that to predict the distance between atoms with a
    # given shortest path length and pair of atom types.
    debug("training null model 2")
    true_distances = []
    shortest_paths = []
    path_dict = dict()
    with torch.no_grad():
        for batch in tqdm(train_data, total=len(train_data)):
            targets = batch.y[:, 0].unsqueeze(1)
            targets_aux = targets.cpu().numpy()
            for i in range(targets_aux.shape[0]):
                true_distances.append(targets_aux[i][0])
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                # Iterate through all pairs of atoms in the molecule and compute the shortest path length
                # as well as the atom types, and concatenate that info to create the dictionary key
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    atom_a = str(mol.GetAtoms()[int(m)].GetSymbol())
                    atom_b = str(mol.GetAtoms()[int(n)].GetSymbol())
                    path_len = len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len)
                    shortest_paths.append(key)
    true_distances = np.array(true_distances)
    for i in range(len(shortest_paths)):
        if shortest_paths[i] in path_dict:
            path_dict[shortest_paths[i]].append(true_distances[i])
        else:
            path_dict[shortest_paths[i]] = [true_distances[i]]
    for key in path_dict:
        path_dict[key] = np.array(path_dict[key]).mean()

    # Test null model 2
    debug("testing null model 2")
    path_lengths = []
    losses = []
    loss_func_aux = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        for batch in tqdm(test_data, total=len(test_data)):
            targets = batch.y[:, 0].unsqueeze(1)
            preds = []
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                # Predict distance based on shortest path length and atom types for each pair of atoms,
                # keeping track of both the predictions and the shortest path lengths
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    atom_a = str(mol.GetAtoms()[int(m)].GetSymbol())
                    atom_b = str(mol.GetAtoms()[int(n)].GetSymbol())
                    path_len = len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1
                    key = ''.join(sorted([atom_a, atom_b])) + str(path_len)
                    preds.append(path_dict[key])
                    path_lengths.append(path_len)
            preds = torch.tensor(np.array(preds)).unsqueeze(1)
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
            loss_aux = loss_aux.cpu().numpy()
            for i in range(loss_aux.shape[0]):
                losses.append(loss_aux[i][0])
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test error avg = {:.4e}".format(loss_avg))

    path_lengths = np.array(path_lengths)
    losses = np.array(losses)
    plt.plot(path_lengths, losses, 'bo', markersize=0.5)
    plt.title("Error vs Path Length")
    plt.ylabel("|True - Predicted|")
    plt.xlabel("Shortest Path")
    plt.savefig(os.path.join(args.save_dir, "null-model-2-error-vs-path"))

    np.save(os.path.join(args.save_dir, "null-model-2-errors"), losses)
