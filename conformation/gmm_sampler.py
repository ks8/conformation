""" Gaussian mixture model. """
import numpy as np
import os
import random
from scipy.stats import multivariate_normal
from typing import List, Union

# noinspection PyPackageRequirements
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    """
    System arguments.
    """
    num_samples: int = 10000  # Number of samples to generate
    max_len: float = 10.  # Maximum distance from origin defining grid
    num_len_partitions: int = 2  # Number of partitions in the grid
    origin: float = 0.0
    num_centers_range: List[int] = [4, 4]  # Range of number of centers
    centers: List = [(0, 0), (0, 10), (10, 0), (10, 10)]
    save_dir: str  # Path to directory containing output files


def to_one_hot(x: int, vals: Union[List, range]) -> List:
    """
    Return a one-hot vector.
    :param x: Data integer.
    :param vals: List of possible data values.
    :return: One-hot vector as list.
    """
    return [int(x == v) for v in vals]


def gmm_sample(centers: List) -> np.ndarray:
    """
    Sample from Gaussian mixture distribution.
    :return:
    """
    center = np.array(centers[random.choice(range(len(centers)))])
    return multivariate_normal(center, np.eye(2)).rvs(1)


def gmm_pdf(x: np.ndarray) -> float:
    """
    PDF.
    :param x:
    :return:
    """
    centers = [(0, 0), (0, 10), (10, 0), (10, 10)]
    pdf = 0
    for i in range(len(centers)):
        pdf += 0.25*multivariate_normal(np.array(centers[i]), np.eye(2)).pdf(x)
    return pdf


def gmm_pdf_generalized(x: np.ndarray, centers: List) -> float:
    """
    PDF generalized.
    :param x:
    :param centers:
    :return:
    """
    pdf = 0
    weight = 1./float(len(centers))
    for i in range(len(centers)):
        pdf += weight*multivariate_normal(np.array(centers[i]), np.eye(2)).pdf(x)
    return pdf


def gmm_sampler(args: Args):
    """
    Sampler.
    :param args:
    :return:
    """
    os.makedirs(args.save_dir)

    counter = 0
    for i in range(10):
        # Define the grid of possible centers
        centers_list = []
        v = np.linspace(args.origin + 2.*i, args.origin + 2.*i + args.max_len, args.num_len_partitions)
        grid = np.meshgrid(v, v)
        for j in range(args.num_len_partitions):
            for k in range(args.num_len_partitions):
                centers_list.append((grid[0][j][k], grid[1][j][k]))
        assert(args.num_centers_range[1] <= len(centers_list))

        for _ in tqdm(range(args.num_samples)):
            num_centers = np.random.choice(range(args.num_centers_range[0], args.num_centers_range[1] + 1))
            center_indices = np.random.choice(range(len(centers_list)), num_centers, replace=False)
            centers = [centers_list[i] for i in center_indices]
            centers.sort()
            sample = gmm_sample(centers)
            np.save(os.path.join(args.save_dir, "gmm_samples_" + str(counter) + ".npy"), sample)

            condition = -1.*np.ones(2*args.num_centers_range[1])
            centers_array = np.array(centers).flatten()
            condition[:len(centers_array)] = centers_array
            condition = np.concatenate((condition, to_one_hot(num_centers - 1, range(args.num_centers_range[1]))))
            np.save(os.path.join(args.save_dir, "gmm_conditions_" + str(counter) + ".npy"), condition)
            counter += 1
