""" Run relational network training. """
import json
from sklearn.model_selection import train_test_split
# noinspection PyPackageRequirements
from tap import Tap
import torch
from torch.optim import Adam
from tqdm import tqdm, trange

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.relational import RelationalNetwork


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata file
    num_epochs: int  # Number of training epochs
    batch_size: int = 10  # Batch size
    lr: float = 1e-4  # Learning rate
    hidden_size: int = 256  # Hidden size
    cuda: bool = False  # Cuda availability


def run_relational_training(args: Args) -> None:
    """
    Run training of relational neural network.
    :param args: System arguments.
    :return: None.
    """

    args.cuda = torch.cuda.is_available()

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

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)
    val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    loss_func = torch.nn.MSELoss(reduction='none')

    model = RelationalNetwork(num_layers=10, num_edge_features=6,
                              num_vertex_features=118)

    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    optimizer = Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in trange(args.num_epochs):
        loss_sum = 0
        for batch in tqdm(train_data, total=len(train_data)):
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()

            model.zero_grad()
            targets = batch.y.unsqueeze(1).cuda()
            # noinspection PyCallingNonCallable
            preds = model(batch)
            loss = loss_func(preds, targets)
            loss = loss.sum() / batch.num_graphs
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()

        loss_avg = loss_sum / len(train_data)
        print('train loss: ', loss_avg)
