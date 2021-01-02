""" Compute 2D KL divergence. """
import numpy as np
from typing import List

from scipy.stats import entropy

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_path_1: str  # Path to one numpy file containing data, which is assumed to have shape (N, 2), where column
    # zero represents the x-axis
    data_path_2: str  # Path to second numpy file containing data, which is assumed to have shape (N, 2), where column
    # zero represents the x-axis
    nx_bins: int = 1000  # Number of histogram bins along the x-axis
    ny_bins: int = 1000  # Number of histogram bins along the y-axis
    x_range: List[float] = [0., 10.]  # Range for histogram x-axis
    y_range: List[float] = [0., 10.]  # Range for histogram y-axis


def compute_kl_divergence(data_path_1: str, data_path_2: str, nx_bins: int = 1000, ny_bins: int = 1000,
                          x_range: List[float] = None, y_range: List[float] = None) -> float:
    """
    Compute the KL divergence between two distributions in 2D.
    :param data_path_1: Path to one numpy file containing data.
    :param data_path_2: Path to second numpy file containing data.
    :param nx_bins: # histogram bins along x-axis
    :param ny_bins: # histogram bins along y-axis
    :param x_range: Range for histogram x-axis
    :param y_range: Range for histogram y-axis
    :return: KL divergence.
    """

    if x_range is None:
        x_range = [0., 10.]
    if y_range is None:
        y_range = [0., 10.]
    data_1 = np.load(data_path_1)
    data_2 = np.load(data_path_2)

    hist_1 = np.histogram2d(data_1[:, 0], data_1[:, 1], bins=[nx_bins, ny_bins],
                            range=[x_range, y_range], density=True)[0].flatten()
    hist_1 = np.where(hist_1 == 0, 1e-10, hist_1)
    hist_1 = np.where(np.isnan(hist_1), 1e-10, hist_1)

    hist_2 = np.histogram2d(data_2[:, 0], data_2[:, 1], bins=[nx_bins, ny_bins],
                            range=[x_range, y_range], density=True)[0].flatten()
    hist_2 = np.where(hist_2 == 0, 1e-10, hist_2)
    hist_2 = np.where(np.isnan(hist_2), 1e-10, hist_2)

    kl = entropy(hist_1, hist_2)

    return kl


def kl_divergence(args: Args) -> None:
    """
    2D KL divergence.
    :param args: System arguments.
    :return: None.
    """
    kl = compute_kl_divergence(args.data_path_1, args.data_path_2, args.nx_bins, args.ny_bins, args.x_range,
                               args.y_range)

    print(f'KL divergence: {kl}')
