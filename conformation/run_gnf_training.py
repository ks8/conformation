""" Run Graph Normalizing Flow training. """
import json
from logging import Logger
import math
import os
from typing import Tuple

from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm, trange

from conformation.dataloader import DataLoader
from conformation.dataset import GraphDataset
from conformation.flows import GRevNet
from conformation.model import build_gnf_model
from conformation.relational import RelationalNetwork
from conformation.sampler import MoleculeSampler
from conformation.train_args_relational import Args
from conformation.utils import save_checkpoint, load_checkpoint, param_count, loss_func


def train(model: nn.Module, optimizer: Adam, data: DataLoader, args: Args, logger: Logger, n_iter: int,
          summary_writer: SummaryWriter) -> Tuple[int, float]:
    """
    Function for training a relational network.
    :param model: Neural network.
    :param optimizer: Adam optimizer.
    :param data: DataLoader.
    :param args: System arguments.
    :param logger: System logger.
    :param n_iter: Total number of iterations.
    :param summary_writer: TensorboardX summary writer.
    :return: total number of iterations.
    """
    # Set up logger
    debug, info = logger.debug, logger.info

    # Run training
    model.train()
    total_loss = 0.0
    loss_sum, batch_count = 0, 0
    for batch in tqdm(data, total=len(data)):

        # TODO: testing!!
        batch.x = torch.randn([batch.x.shape[0], args.hidden_size])
        batch.edge_attr = torch.randn([batch.edge_attr.shape[0], args.hidden_size])

        # Move batch to cuda
        if args.cuda:
            batch.x = batch.x.cuda()
            batch.edge_attr = batch.edge_attr.cuda()

        # Zero gradients
        model.zero_grad()

        print(model(batch))
        exit()

    return n_iter, total_loss


def run_gnf_training(args: Args, logger: Logger) -> None:
    """
    Run training of graph normalizing flow.
    :param args: System arguments.
    :param logger: Logging.
    :return: None.
    """

    # Save directories
    os.makedirs(os.path.join(args.save_dir, "checkpoints"))

    # Set up logger
    debug, info = logger.debug, logger.info

    # Print args
    debug(args)

    # Check cuda availability
    args.cuda = torch.cuda.is_available()

    # Load data
    debug("loading data")
    metadata = json.load(open(args.data_path))
    train_metadata, remaining_metadata = train_test_split(metadata, test_size=0.2, random_state=0)
    validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=0)

    # Convert to dataset
    train_data = GraphDataset(train_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                              max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                              partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                              valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                              hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                              charge_types=args.charge_types, formal_charge=args.formal_charge,
                              r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                              default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                              chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                              hybridization_types=args.hybridization_types, chi_types=args.chi_types,
                              improved_architecture=args.improved_architecture, degree_types=args.degree_types,
                              degree=args.degree, num_hydrogen_types=args.num_hydrogen_types,
                              num_hydrogen=args.num_hydrogen,
                              num_radical_electron_types=args.num_radical_electron_types,
                              num_radical_electron=args.num_radical_electron, conjugated=args.conjugated,
                              bond_type=args.bond_type, bond_ring=args.bond_ring, bond_stereo=args.bond_stereo,
                              bond_stereo_types=args.bond_stereo_types, shortest_path=args.shortest_path,
                              same_ring=args.same_ring, autoencoder=args.autoencoder)
    val_data = GraphDataset(validation_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                            max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                            partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                            valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                            hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                            charge_types=args.charge_types, formal_charge=args.formal_charge,
                            r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                            default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                            chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                            hybridization_types=args.hybridization_types, chi_types=args.chi_types,
                            improved_architecture=args.improved_architecture, degree_types=args.degree_types,
                            degree=args.degree, num_hydrogen_types=args.num_hydrogen_types,
                            num_hydrogen=args.num_hydrogen,
                            num_radical_electron_types=args.num_radical_electron_types,
                            num_radical_electron=args.num_radical_electron, conjugated=args.conjugated,
                            bond_type=args.bond_type, bond_ring=args.bond_ring, bond_stereo=args.bond_stereo,
                            bond_stereo_types=args.bond_stereo_types, shortest_path=args.shortest_path,
                            same_ring=args.same_ring, autoencoder=args.autoencoder)
    test_data = GraphDataset(test_metadata, atom_types=args.atom_types, bond_types=args.bond_types,
                             max_path_length=args.max_shortest_path_length, atomic_num=args.atomic_num,
                             partial_charge=args.partial_charge, mmff_atom_types_one_hot=args.mmff_atom_types_one_hot,
                             valence_types=args.valence_types, valence=args.valence, aromatic=args.aromatic,
                             hybridization=args.hybridization, assign_stereo=args.assign_stereo,
                             charge_types=args.charge_types, formal_charge=args.formal_charge,
                             r_covalent=args.r_covalent, r_vanderwals=args.r_vanderwals,
                             default_valence=args.default_valence, max_ring_size=args.max_ring_size, rings=args.rings,
                             chirality=args.chirality, mmff94_atom_types=args.mmff94_atom_types,
                             hybridization_types=args.hybridization_types, chi_types=args.chi_types,
                             improved_architecture=args.improved_architecture, degree_types=args.degree_types,
                             degree=args.degree, num_hydrogen_types=args.num_hydrogen_types,
                             num_hydrogen=args.num_hydrogen,
                             num_radical_electron_types=args.num_radical_electron_types,
                             num_radical_electron=args.num_radical_electron, conjugated=args.conjugated,
                             bond_type=args.bond_type, bond_ring=args.bond_ring, bond_stereo=args.bond_stereo,
                             bond_stereo_types=args.bond_stereo_types, shortest_path=args.shortest_path,
                             same_ring=args.same_ring, autoencoder=args.autoencoder)

    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug(f'train size = {train_data_length:,} | val size = {val_data_length:,} | test size = {test_data_length:,}'
          )

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size, shuffle=False,
                            sampler=MoleculeSampler(train_data, args.random_sample))

    # Load/build model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.cuda, args.gpu_device)
    else:
        args.num_edge_features = 0
        args.num_vertex_features = 0
        if args.bond_type:
            args.num_edge_features += len(args.bond_types) + 1
        if args.conjugated:
            args.num_edge_features += 1
        if args.bond_ring:
            args.num_edge_features += 1
        if args.bond_stereo:
            args.num_edge_features += len(args.bond_stereo_types)
        if args.shortest_path:
            args.num_edge_features += args.max_shortest_path_length
        if args.same_ring:
            args.num_edge_features += 1
        if args.autoencoder:
            args.num_edge_features += 1
        if args.atomic_num:
            args.num_vertex_features += len(args.atom_types)
        if args.valence:
            args.num_vertex_features += len(args.valence_types)
        if args.partial_charge:
            args.num_vertex_features += 1
        if args.mmff_atom_types_one_hot:
            args.num_vertex_features += len(args.mmff94_atom_types)
        if args.aromatic:
            args.num_vertex_features += 1
        if args.hybridization:
            args.num_vertex_features += len(args.hybridization_types)
        if args.formal_charge:
            args.num_vertex_features += len(args.charge_types)
        if args.r_covalent:
            args.num_vertex_features += 1
        if args.r_vanderwals:
            args.num_vertex_features += 1
        if args.default_valence:
            args.num_vertex_features += len(args.valence_types)
        if args.rings:
            args.num_vertex_features += args.max_ring_size - 2
        if args.chirality:
            args.num_vertex_features += len(args.chi_types)
        if args.degree:
            args.num_vertex_features += len(args.degree_types)
        if args.num_hydrogen:
            args.num_vertex_features += len(args.num_hydrogen_types)
        if args.num_radical_electron:
            args.num_vertex_features += len(args.num_radical_electron_types)

        debug('Building model')
        model = build_gnf_model(args)
        # mask = 0
        # s = RelationalNetwork(int(args.hidden_size/2), 1, int(args.hidden_size/2), int(args.hidden_size/2),
        #                       args.final_linear_size, args.final_output_size, gnf=True)
        #
        # t = RelationalNetwork(int(args.hidden_size/2), 1, int(args.hidden_size/2), int(args.hidden_size/2),
        #                       args.final_linear_size, args.final_output_size, gnf=True)
        #
        # model = GRevNet(s, t, mask)

    # Print model info
    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    # Move model to cuda if available
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Loss func and optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Run training
    summary_writer = SummaryWriter(logdir=args.save_dir)
    best_epoch, n_iter = 0, 0
    best_loss = float('inf')
    for epoch in trange(args.num_epochs):
        n_iter, total_loss = train(model, optimizer, train_data, args, logger, n_iter, summary_writer)
        debug(f"Epoch {epoch} total loss = {total_loss:.4e}")
        summary_writer.add_scalar("Total Train Loss", total_loss, epoch)
        save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
        if total_loss < best_loss:
            save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'best.pt'))
            best_loss = total_loss
            best_epoch = epoch

    debug(f"Best epoch: {best_epoch} with total loss = {best_loss:.4e}")
