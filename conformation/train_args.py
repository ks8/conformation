""" Args class for training arguments. """
# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata JSON file
    save_dir: str  # Path to directory containing output files (average distances)
    checkpoint_path: str = None  # Path to checkpoint file
    conditional: bool = False  # Whether or not to run a conditional normalizing flow
    graph_model_path: str = None  # Path to saved graph model checkpoint file
    num_layers: int = 10  # Number of RealNVP layers
    num_epochs: int = 10  # Number of training epochs
    batch_size: int = 10  # Training batch size
    lr: float = 1e-4  # Learning rate
    input_dim: int = 28  # Number of pairwise distances (dimensionality of input vectors)
    condition_dim: int = 256  # Hidden size of condition vectors
    num_atoms: int = 8  # Number of atoms in molecule
    hidden_size: int = 256  # Hidden size
    log_frequency: int = 10  # Log frequency
    cuda: bool = False  # Cuda availability
