""" Args class for training arguments. """
from typing_extensions import Literal

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path: str  # Path to metadata JSON file
    save_dir: str  # Path to directory containing output files
    checkpoint_path: str = None  # Path to PyTorch checkpoint file
    conditional_base: bool = False  # Whether or not to run a conditional base normalizing flow
    conditional_concat: bool = False  # Whether or not to run a conditional concat normalizing flow
    covariance_factor: float = 1.0  # Multiplicative factor for the base distribution covariance matrix
    graph_model_path: str = None  # Path to saved graph model checkpoint file (conditional_base = True and using
    # molecule data)
    num_layers: int = 10  # Number of flow layers (2 layers is equivalent to one full coupling layer)
    num_epochs: int = 10  # Number of training epochs
    batch_size: int = 10  # Training batch size
    lr: float = 1e-4  # Learning rate
    input_dim: int = 28  # Dimensionality of input vectors that pass through the flow
    condition_dim: int = 256  # Length of condition vectors, or hidden size if they are multi-dimensional
    # (conditional_base = True or conditional_concat = True)
    base_output_dim: int = 1  # Output dimension for the feedforward network that produces the mean vector for the base
    #  distribution. Use base_output_dim=1 for previous molecule work, otherwise base_output_dim should be set to
    #  input_dim (conditional_base = True)
    hidden_size: int = 256  # Hidden size for flow layers
    num_internal_layers: int = 3  # Number of linear layers for the neural networks that comprise each flow layer
    base_hidden_size: int = 1024  # Hidden size for feedforward network that produces the mean vector for the base
    # distribution (conditional_base = True)
    log_frequency: int = 10  # Log frequency
    cuda: bool = False  # Cuda availability (this is set automatically)
    padding: bool = False  # Whether or not padding will be used (conditional_base = True).
    s_output_activation: Literal["tanh", "lrelu"] = "tanh"  # Which output activation function to use for the "s"
    # RealNVP neural network output activation. Default is nn.tanh, other option is nn.LeakyReLU.
