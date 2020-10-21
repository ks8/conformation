""" RMSD pipeline: run RMSD pruning on a set of conformation files. """
import os

# List of directories containing conformations that need to be pruned.
conformation_directories = ["hmc-tryptophan-test", "parallel-tryptophan-test"]

# File name of conformations (assumed to be common across all directories)
file_name = "accepted-conformations.bin"

# System arguments:
# data_path: str  # Path to RDKit binary file containing conformations
# minimize: bool = False  # Whether or not to minimize conformations before RMSD pruning
# rmsd_func: Literal["GetBestRMS", "AlignMol"] = "GetBestRMS"  # RMSD computation options
# remove_Hs: bool = False  # Whether or not to do RMSD calculations without Hydrogen atoms
# energy_threshold: float = 2.0  # Energy threshold above which 2 conformers are considered different (kcal/mol)
# rmsd_threshold: float = 0.5  # RMSD threshold for deciding two conformers are the same (Angstroms)
# save_dir: str  # Path to output file containing pruned conformations


def run_rmsd_pipeline() -> None:
    """
    RMSD pipeline.
    :return: None.
    """
    for dir_path in conformation_directories:
        f = os.path.join(dir_path, file_name)
        os.system(f'python run_rmsd_pruning.py --data_path={f} --minimize --remove_Hs --save_dir={dir_path}')


if __name__ == '__main__':
    run_rmsd_pipeline()
