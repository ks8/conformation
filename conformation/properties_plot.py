""" Plot distributions of conformation properties. """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

# noinspection PyPackageRequirements
from tap import Tap


class Args(Tap):
    """
    System arguments.
    """
    data_dir: str = None  # Path to directory containing input files
    num_energy_bins: int = 40  # Number of energy histogram bins
    num_rms_bins: int = 20  # Number of dihedral histogram bins
    num_dihedral_bins: int = 20  # Number of dihedral histogram bins
    save_prefix: str = None  # File name for plot image
    upper_energy_bound: float = 5.0  # Upper bound for molecule energy
    lower_energy_bound: float = -5.0  # Lower bound for molecule energy
    energy: bool = False  # Use when plotting energy values
    rms: bool = False  # Use when plotting rms values
    dihedral: bool = False  # Use when plotting dihedral angle values


def properties_plot(args: Args) -> None:
    """
    Plot histograms of energies, RMS, dihedrals for molecular conformations.
    :param args: Argparse arguments.
    :return: None.
    """
    energies = []
    rms_list = []
    dihedral_list = []
    for _, _, files in os.walk(args.data_dir):
        for f in files:
            if f[:10] == "energy-rms":
                file = open(os.path.join(args.data_dir, f))
                contents = file.readlines()
                energy = float(contents[0].split()[1])
                rms = float(contents[1].split()[1])
                dihedral = float(contents[2].split()[1])
                if args.lower_energy_bound < energy < args.upper_energy_bound:
                    energies.append(energy)
                    rms_list.append(rms)
                    dihedral_list.append(dihedral*(180.0/math.pi))

    if args.energy:
        plt.hist(energies, bins=args.num_energy_bins)
        plt.title("Ethane ETKDG Energies")
        plt.ylabel("Frequency")
        plt.xlabel("Energy (kcal/mol)")
        plt.savefig(args.save_prefix + "-energies.png")
        plt.clf()

    if args.rms:
        plt.hist(rms_list, bins=args.num_rms_bins)
        plt.title("Ethane RMS values")
        plt.ylabel("Frequency")
        plt.xlabel("Distance ($\AA$)")
        plt.savefig(args.save_prefix + "-rms.png")
        plt.clf()

    if args.dihedral:
        plt.hist(dihedral_list, bins=args.num_rms_bins)
        plt.title("Ethane Dihedral Values")
        plt.ylabel("Frequency")
        plt.xlabel("Angle (Degrees)")
        plt.savefig(args.save_prefix + "-dihedral.png")
        plt.clf()

    if args.energy and args.dihedral:
        plt.plot(dihedral_list, energies, 'bo')
        plt.title("Ethane Energy vs Dihedral")
        plt.ylabel("Energy (kcal/mol)")
        plt.xlabel("Dihedral Angle (Degrees)")
        plt.savefig(args.save_prefix + "-energy-vs-dihedral.png")
        plt.clf()
