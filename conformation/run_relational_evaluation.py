""" Run relational network training. """
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
    test_data = DataLoader(test_data, args.batch_size)

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

    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(test_data, total=len(test_data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            targets = batch.y.unsqueeze(1).cuda()
            preds = model(batch)
            loss = loss_func(preds, targets)
            loss = torch.sqrt_(loss)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test loss avg = {:.4e}".format(loss_avg))
