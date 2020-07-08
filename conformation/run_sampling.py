""" Run neural network evaluation/sampling. """
import argparse
from argparse import Namespace
import os
from pprint import pformat

import torch

from conformation.sample import sample
from conformation.utils import load_checkpoint, param_count


def run_sampling(args: Namespace) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :return: None.
    """

    print(pformat(vars(args)))

    # Load model
    if args.checkpoint_path is not None:
        print('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.save_dir, args.cuda)

        print(model)
        print('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            print('Moving model to cuda')
            model = model.cuda()

        sample(model, args.smiles, args.save_dir, args.num_atoms, args.offset, args.num_layers, args.num_samples,
               args.dihedral, args.dihedral_vals)

    else:
        print('Must specify a model to load.')
        exit()


def main():
    """
    Parse arguments and run run_training function.
    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_atoms', type=int, dest='num_atoms', default=8, help='Number of atoms')
    parser.add_argument('--num_layers', type=int, dest='num_layers', default=6, help='# RealNVP layers')
    parser.add_argument('--num_samples', type=int, dest='num_samples', default=10000, help='# test samples')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default=None, help='Save directory')
    parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path',
                        default=None, help='Directory of checkpoint')
    parser.add_argument('--smiles', type=str, dest='smiles',
                        default=None, help='Molecular SMILES string')
    parser.add_argument('--offset', type=float, dest='offset',
                        default=0.0005, help='Distance bounds matrix offset')
    parser.add_argument('--dihedral', action='store_true', default=False,
                        help='Use when computing dihedral angle values')
    parser.add_argument('--dihedral_vals', type=int, dest='dihedral_vals', nargs='+', default=[2, 0, 1, 5],
                        help='Atom IDs for dihedral')
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    args.cuda = torch.cuda.is_available()

    run_sampling(args)


if __name__ == '__main__':
    main()
