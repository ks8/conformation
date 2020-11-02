""" Analyze distributions pipeline: run analyze_distributions.py on a set of conformation files. """
import os

# List of directories containing conformations that need to be pruned.
conformation_directories = ["hmc-tryptophan-test", "parallel-tryptophan-test"]

# File name of conformations (assumed to be common across all directories)
file_name = "all-conformations.bin"

# System arguments
# data_path: str  # Path to RDKit binary file containing conformations
# num_energy_decimals: int = 3  # Number of energy decimals used for computing empirical minimized energy probability
# subsample_frequency: int = 1  # Frequency at which to compute sample information
# save_dir: str  # Path to directory containing output files


def run_analyze_distributions() -> None:
    """
    RMSD pipeline.
    :return: None.
    """
    for dir_path in conformation_directories:
        f = os.path.join(dir_path, file_name)
        os.system(f'python analyze_distributions.py --data_path={f} --save_dir={dir_path}')


if __name__ == '__main__':
    run_analyze_distributions()
