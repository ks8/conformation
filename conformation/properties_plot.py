import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math


def energy_plot(args):
    """
    Plot histograms of energies and RMS for molecular conformations
    :param args: Argparse arguments
    :return: None
    """
    energies = []
    rms_list = []
    dihedral_list = []
    for _, _, files in os.walk(args.input):
        for f in files:
            if f[:10] == "energy-rms":
                file = open(os.path.join(args.input, f))
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
        plt.savefig(args.out + "-energies.png")
        plt.clf()

    if args.rms:
        plt.hist(rms_list, bins=args.num_rms_bins)
        plt.title("Ethane RMS values")
        plt.ylabel("Frequency")
        plt.xlabel("Distance ($\AA$)")
        plt.savefig(args.out + "-rms.png")
        plt.clf()

    if args.dihedral:
        plt.hist(dihedral_list, bins=args.num_rms_bins)
        plt.title("Ethane Dihedral Values")
        plt.ylabel("Frequency")
        plt.xlabel("Angle (Degrees)")
        plt.savefig(args.out + "-dihedral.png")
        plt.clf()

    if args.energy and args.dihedral:
        plt.plot(dihedral_list, energies, 'bo')
        plt.title("Ethane Energy vs Dihedral")
        plt.ylabel("Energy (kcal/mol)")
        plt.xlabel("Dihedral Angle (Degrees)")
        plt.savefig(args.out + "-energy-vs-dihedral.png")
        plt.clf()


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None, help='Folder path containing input files')
    parser.add_argument('--num_energy_bins', type=int, dest='num_energy_bins', default=40,
                        help='# energy histogram bins')
    parser.add_argument('--num_rms_bins', type=int, dest='num_rms_bins', default=20, help='# rms histogram bins')
    parser.add_argument('--num_dihedral_bins', type=int, dest='num_dihedral_bins', default=20,
                        help='# dihedral histogram bins')
    parser.add_argument('--out', type=str, dest='out', default=None, help='File name for plot image')
    parser.add_argument('--upper_energy_bound', type=float, dest='upper_energy_bound', default=5.0,
                        help='Upper bound for molecule energy')
    parser.add_argument('--lower_energy_bound', type=float, dest='lower_energy_bound', default=-5.0,
                        help='Lower bound for molecule energy')
    parser.add_argument('--energy', action='store_true', default=False,
                        help='Use when plotting energy values')
    parser.add_argument('--rms', action='store_true', default=False,
                        help='Use when plotting rms values')
    parser.add_argument('--dihedral', action='store_true', default=False,
                        help='Use when plotting dihedral angle values')
    args = parser.parse_args()

    energy_plot(args)


if __name__ == '__main__':
    main()
