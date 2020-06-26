import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math


def dihedral_plot(args):
    """
    Plot histograms of energies and RMS for molecular conformations
    :param args: Argparse arguments
    :return: None
    """
    dihedral_list = []
    for _, _, files in os.walk(args.folder):
        for f in files:
            if f[:10] == "energy-rms":
                file = open(os.path.join(args.folder, f))
                contents = file.readlines()
                dihedral = float(contents[2].split()[1])
                dihedral_list.append(dihedral*(180.0/math.pi))

    plt.clf()
    plt.hist(dihedral_list, bins=args.num_dihedral_bins)
    plt.title("Ethane Dihedral Values")
    plt.ylabel("Frequency")
    plt.xlabel("Angle (Degrees)")
    plt.savefig(args.out + "-dihedral.png")


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--num_dihedral_bins', type=int, dest='num_dihedral_bins', default=20,
                        help='# dihedral histogram bins')
    parser.add_argument('--out', type=str, dest='out', default=None, help='File name for plot image')
    args = parser.parse_args()

    dihedral_plot(args)


if __name__ == '__main__':
    main()
