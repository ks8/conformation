""" Null model for predicting pairwise distances based on minimum path length between pairs of atoms. """
import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

# noinspection PyUnresolvedReferences
from rdkit import Chem
# noinspection PyUnresolvedReferences
from rdkit.Chem import AllChem, rdMolTransforms, rdmolops
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str  # Path to data directory
    data_path: str  # Path to metadata file
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    num_edge_features: int = 6  # Number of edge features
    num_vertex_features: int = 118  # Number of vertex features
    cuda: bool = False  # Cuda availability
    log_frequency: int = 10  # Log frequency


def null_model(args: Args) -> None:
    """
    Predict atomic distances for a molecule based on minimum path length between pairs of atoms.
    :param args: System arguments.
    :return: None.
    """
    avg_distances = []
    for _, _, files in os.walk(args.data_dir):
        for f in files:
            distances = np.load(os.path.join(args.data_dir, f))
            avg_distances.append(np.mean(distances))
    avg_distances = np.array(avg_distances)
    avg_distance = avg_distances.mean()

    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    print("loading data")
    train_data = GraphDataset(train_metadata)
    val_data = GraphDataset(validation_metadata)
    test_data = GraphDataset(test_metadata)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    print('train size = {:,} | val size = {:,} | test size = {:,}'.format(
        train_data_length,
        val_data_length,
        test_data_length)
    )

    # Convert to iterator
    train_data = DataLoader(train_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()

    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        for batch in tqdm(test_data, total=len(test_data)):
            targets = batch.y.unsqueeze(1).cuda()
            preds = torch.zeros_like(targets)
            for i in range(preds.shape[0]):
                preds[i] = avg_distance
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        print("Test loss avg = {:.4e}".format(loss_avg))

    #################################################
    true_distances = []
    uid_dict = pickle.load(open("metadata-qm9-2-nmin-20-nmax-1-RDKitinit-10000-MDsteps/uid_dict.p", "rb"))
    shortest_paths = []
    path_dict = dict()
    with torch.no_grad():
        for batch in tqdm(train_data, total=len(train_data)):
            targets = batch.y.unsqueeze(1).cuda()
            targets_aux = targets.cpu().numpy()
            for i in range(targets_aux.shape[0]):
                true_distances.append(targets_aux[i][0])
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    shortest_paths.append(len(rdmolops.GetShortestPath(mol, int(m), int(n))))
    true_distances = np.array(true_distances)
    shortest_paths = np.array(shortest_paths)
    for i in range(shortest_paths.shape[0]):
        if shortest_paths[i] in path_dict:
            path_dict[shortest_paths[i]].append(true_distances[i])
        else:
            path_dict[shortest_paths[i]] = []
            path_dict[shortest_paths[i]].append(true_distances[i])
    for key in path_dict:
        path_dict[key] = np.array(path_dict[key]).mean()

    path_lengths = []
    losses = []
    loss_func_aux = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        for batch in tqdm(test_data, total=len(test_data)):
            targets = batch.y.unsqueeze(1).cuda()
            preds = []
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    preds.append(path_dict[len(rdmolops.GetShortestPath(mol, int(m), int(n)))])
                    path_lengths.append(path_dict[len(rdmolops.GetShortestPath(mol, int(m), int(n)))])
            preds = torch.tensor(np.array(preds)).unsqueeze(1)
            preds = preds.cuda()
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
            loss_aux = loss_aux.cpu().numpy()
            for i in range(loss_aux.shape[0]):
                losses.append(loss_aux[i][0])
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        print("Test loss avg = {:.4e}".format(loss_avg))

        path_lengths = np.array(path_lengths)
        losses = np.array(losses)
        plt.plot(path_lengths, losses, 'bo', markersize=0.5)
        plt.title("Error vs Path Length")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("Shortest Path")
        plt.savefig("qm9-path-length-model-error-vs-path-test-set")
        plt.clf()
        plt.plot(np.log(path_lengths), np.log(losses), 'bo', markersize=0.5)
        plt.title("Error vs Atomic Pairwise Shortest Paths")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("Shortest Path")
        plt.savefig("qm9-path-length-model-error-vs-path-log-test-set")





