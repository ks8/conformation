""" Run relational network evaluation. """
import itertools
import json
from logging import Logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from typing_extensions import Literal

from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.run_relational_training import Args as RelationalTrainArgs
from conformation.utils import param_count


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    uid_path: str  # Path to uid dictionary
    checkpoint_path: str  # Directory of checkpoint to load saved model
    save_dir: str  # Directory for logger
    batch_size: int = 10  # Batch size
    dataset: Literal["train", "val", "test"] = "test"  # Which dataset to evaluate
    cuda: bool = False  # Cuda availability
    distance_analysis: bool = False  # Whether or not to print selective bond/edge information
    distance_analysis_lo: float = 1.0  # Lower distance analysis bound
    distance_analysis_hi: float = 2.0  # Upper distance analysis bound


def run_relational_evaluation(args: Args, logger: Logger) -> None:
    """
    Run evaluation of relational neural network.
    :param args: System arguments.
    :param logger: Logging.
    :return: None.
    """

    # Set up logger
    debug, info = logger.debug, logger.info

    # Print args
    debug(args)

    # Check cuda availability
    args.cuda = torch.cuda.is_available()

    # Load metadata
    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    # Create datasets
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
    val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Load/build model
    debug('Loading model from {}'.format(args.checkpoint_path))
    state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    loaded_args = RelationalTrainArgs().from_dict(state['args'])
    loaded_state_dict = state['state_dict']
    model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                              loaded_args.num_vertex_features, loaded_args.final_linear_size,
                              loaded_args.final_output_size)
    model.load_state_dict(loaded_state_dict)

    # Print model details
    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Loss function
    loss_func = torch.nn.MSELoss()

    # Error function
    loss_func_aux = torch.nn.MSELoss(reduction='none')

    # Load uid-smiles dictionary
    uid_dict = pickle.load(open(args.uid_path, "rb"))

    # Select dataset to evaluate
    if args.dataset == "test":
        data = test_data
    elif args.dataset == "train":
        data = train_data
    else:
        data = val_data

    errors = []
    true_distances = []
    shortest_paths = []
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(data, total=len(data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            if loaded_args.final_output_size == 2:
                targets = batch.y.cuda()
            else:
                targets = batch.y.cuda()[:, 0].unsqueeze(1)
            preds = model(batch)

            # Compute error, target (true distance), and shortest path for each edge
            # Compute error for each edge and append to errors list
            loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
            loss_aux = loss_aux.cpu().numpy()
            for i in range(loss_aux.shape[0]):
                errors.append(loss_aux[i][0])

            # Compute true distance for each edge and append to true_distances list
            if args.distance_analysis:
                candidates = []
            targets_aux = targets.cpu().numpy()
            for i in range(targets_aux.shape[0]):
                edge_distance = targets_aux[i][0]
                true_distances.append(edge_distance)
                if args.distance_analysis:
                    if args.distance_analysis_lo < edge_distance < args.distance_analysis_hi:
                        candidates.append(i)

            # Compute shortest path length for each edge and append to shortest_paths list
            if args.distance_analysis:
                j = 0
            for i in range(batch.uid.shape[0]):
                uid = batch.uid[i].item()
                smiles = uid_dict[uid]
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)
                for m, n in itertools.combinations(list(np.arange(mol.GetNumAtoms())), 2):
                    if args.distance_analysis:
                        if j in candidates:
                            debug(str(mol.GetAtoms()[int(m)].GetSymbol()) + " " +
                                  str(mol.GetAtoms()[int(n)].GetSymbol()) + " " +
                                  str(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1))
                        j += 1
                    shortest_paths.append(len(rdmolops.GetShortestPath(mol, int(m), int(n))) - 1)

            # Compute RMSE loss
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1

        loss_avg = loss_sum / batch_count
        debug("Test loss avg = {:.4e}".format(loss_avg))

        # Convert to numpy
        errors = np.array(errors)
        true_distances = np.array(true_distances)
        shortest_paths = np.array(shortest_paths)

        # Plotting
        # Plot error vs true distance
        plt.plot(true_distances, errors, 'bo', markersize=0.5)
        plt.title("Error vs True Atomic Pairwise Distances")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("True")
        plt.savefig(os.path.join(args.save_dir, "error-vs-distance"))
        plt.clf()

        # Plot log error vs log true distance
        plt.plot(np.log(true_distances), np.log(errors), 'bo', markersize=0.5)
        plt.title("Error vs True Atomic Pairwise Distances")
        plt.ylabel("log(|True - Predicted|)")
        plt.xlabel("log(True)")
        plt.savefig(os.path.join(args.save_dir, "log-error-vs-log-distance"))
        plt.clf()

        # Plot error vs shortest path
        plt.plot(shortest_paths, errors, 'bo', markersize=0.5)
        plt.title("Error vs Atomic Pairwise Shortest Paths")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("Shortest Path")
        plt.savefig(os.path.join(args.save_dir, "error-vs-path-length"))
        plt.clf()

        # Plot shortest path vs distance
        plt.plot(true_distances, shortest_paths, 'bo', markersize=0.5)
        plt.title("Error vs Atomic Pairwise Shortest Paths")
        plt.ylabel("Shortest Path")
        plt.xlabel("True Distance")
        plt.savefig(os.path.join(args.save_dir, "path-length-vs-distance"))
        plt.clf()
