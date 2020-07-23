""" Run relational network training. """
###
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import rdmolops
###
from logging import Logger
import json
import os

from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from tqdm import tqdm

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.utils import param_count


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    num_edge_features: int = 6  # Number of edge features
    num_vertex_features: int = 118  # Number of vertex features
    cuda: bool = False  # Cuda availability
    checkpoint_path: str  # Directory of checkpoint to load saved model
    save_dir: str  # Save directory
    log_frequency: int = 10  # Log frequency


def run_relational_evaluation(args: Args, logger: Logger) -> None:
    """
    Run evaluation of relational neural network.
    :param args: System arguments.
    :param logger: Logging.
    :return: None.
    """

    # Save directories
    os.makedirs(os.path.join(args.save_dir, "checkpoints"))

    # Set up logger
    debug, info = logger.debug, logger.info

    debug(args)

    args.cuda = torch.cuda.is_available()

    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    debug("loading data")
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
    test_data = DataLoader(train_data, args.batch_size)

    # Load/build model
    debug('Loading model from {}'.format(args.checkpoint_path))

    # Load model and args
    state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
    loaded_args = Args().from_dict(state['args'])
    loaded_state_dict = state['state_dict']

    model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                              loaded_args.num_vertex_features, loaded_args.final_linear_size)
    model.load_state_dict(loaded_state_dict)

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()

    ###
    loss_func_aux = torch.nn.MSELoss(reduction='none')
    losses = []
    true_distances = []
    shortest_paths = []
    uid_dict = pickle.load(open("metadata-qm9-2-nmin-20-nmax-1-RDKitinit-10000-MDsteps/uid_dict.p", "rb"))
    ###
    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(test_data, total=len(test_data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            targets = batch.y.unsqueeze(1).cuda()
            preds = model(batch)
            loss = loss_func(preds, targets)
            ###
            loss_aux = torch.sqrt_(loss_func_aux(preds, targets))
            loss_aux = loss_aux.cpu().numpy()
            for i in range(loss_aux.shape[0]):
                losses.append(loss_aux[i][0])
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
            ###
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test loss avg = {:.4e}".format(loss_avg))

        ###
        losses = np.array(losses)
        true_distances = np.array(true_distances)
        shortest_paths = np.array(shortest_paths)
        plt.plot(true_distances, losses, 'bo', markersize=0.5)
        plt.title("Error vs True Atomic Pairwise Distances")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("True")
        plt.savefig("qm9-2-nmin-20-nmax-100-epochs-10-layers-error-vs-distance-train-set-2")
        plt.clf()
        plt.plot(np.log(true_distances), np.log(losses), 'bo', markersize=0.5)
        plt.title("Error vs True Atomic Pairwise Distances")
        plt.ylabel("log(|True - Predicted|)")
        plt.xlabel("log(True)")
        plt.savefig("qm9-2-nmin-20-nmax-100-epochs-10-layers-error-vs-distance-log-train-set-2")
        plt.clf()
        plt.plot(shortest_paths, losses, 'bo', markersize=0.5)
        plt.title("Error vs Atomic Pairwise Shortest Paths")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("Shortest Path")
        plt.savefig("qm9-2-nmin-20-nmax-100-epochs-10-layers-error-vs-path-train-set-2")
        plt.clf()
        plt.plot(np.log(shortest_paths), np.log(losses), 'bo', markersize=0.5)
        plt.title("Error vs Atomic Pairwise Shortest Paths")
        plt.ylabel("|True - Predicted|")
        plt.xlabel("Shortest Path")
        plt.savefig("qm9-2-nmin-20-nmax-100-epochs-10-layers-error-vs-path-log-train-set-2")
        ###
