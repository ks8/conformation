""" Run relational network training. """
from logging import Logger
import json
import os

from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork
from conformation.utils import param_count


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    num_epochs: int  # Number of training epochs
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    hidden_size: int = 256  # Hidden size
    num_layers: int = 10  # Number of layers
    num_edge_features: int = 6  # Number of edge features
    final_linear_size: int = 1024  # Size of last linear layer
    num_vertex_features: int = 118  # Number of vertex features
    cuda: bool = False  # Cuda availability
    checkpoint_path: str = None  # Directory of checkpoint to load saved model
    save_dir: str  # Save directory
    log_frequency: int = 10  # Log frequency


def run_relational_training(args: Args, logger: Logger) -> None:
    """
    Run training of relational neural network.
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
    debug(f'train size = {train_data_length:,} | val size = {val_data_length:,} | test size = {test_data_length:,}'
          )

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)
    # val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Load/build model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        # Load model and args
        state = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
        loaded_args = Args()
        loaded_args.from_dict(state['args'])
        loaded_state_dict = state['state_dict']

        model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                                  loaded_args.num_vertex_features, loaded_args.final_linear_size)
        model.load_state_dict(loaded_state_dict)
    else:
        debug('Building model')
        model = RelationalNetwork(args.hidden_size, args.num_layers, args.num_edge_features, args.num_vertex_features,
                                  args.final_linear_size)

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Loss func and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()

    n_iter = 0
    for epoch in trange(args.num_epochs):
        loss_sum, batch_count = 0, 0
        for batch in tqdm(train_data, total=len(train_data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()

            model.zero_grad()
            targets = batch.y.unsqueeze(1).cuda()
            # noinspection PyCallingNonCallable
            preds = model(batch)
            loss = loss_func(preds, targets)
            loss_sum += loss.item()
            batch_count += 1
            n_iter += batch.num_graphs

            loss.backward()
            optimizer.step()

            if (n_iter // args.batch_size) % args.log_frequency == 0:
                loss_avg = loss_sum / batch_count
                loss_sum, batch_count = 0, 0
                debug("Loss avg = {:.4e}".format(loss_avg))

        state = {
            'args': args.as_dict(),
            'state_dict': model.state_dict()
        }
        torch.save(state, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))

    with torch.no_grad():
        loss_sum, batch_count = 0, 0
        model.eval()
        for batch in tqdm(test_data, total=len(test_data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()
            targets = batch.y.unsqueeze(1).cuda()
            preds = model(batch)
            loss = loss_func(preds, targets)
            loss_sum += loss.item()
            batch_count += 1
        loss_avg = loss_sum / batch_count
        debug("Test loss avg = {:.4e}".format(loss_avg))
