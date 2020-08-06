""" Relational neural network auxiliary functions. """
import torch
import torch.nn as nn
from typing import Tuple

from conformation.relational import RelationalNetwork
from conformation.train_args_relational import Args as RelationalTrainArgs


def load_relational_checkpoint(path: str, args: RelationalTrainArgs) -> Tuple[nn.Module, RelationalTrainArgs]:
    """
    Loads a relational network model checkpoint.
    :param path: Path to checkpoint file.
    :param args: Relational training args.
    :return: Loaded model.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_args = args.from_dict(state['args'])
    loaded_state_dict = state['state_dict']
    model = RelationalNetwork(loaded_args.hidden_size, loaded_args.num_layers, loaded_args.num_edge_features,
                              loaded_args.num_vertex_features, loaded_args.final_linear_size,
                              loaded_args.final_output_size)
    model.load_state_dict(loaded_state_dict)

    return model, loaded_args
