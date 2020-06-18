import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def energy_plot(args):
    energies = []
    rms = []
    for _, _, files in os.walk(args.folder):
        for f in files:
            if f[:10] == "energy-rms":
                file = open(os.path.join(args.folder, f))
                energy = file.readlines()
                if -5 < float(energy[0].split()[1]) < 5:
                    energies.append(float(energy[0].split()[1]))
                else:
                    print(float(energy[0].split()[1]))
                rms.append(float(energy[1].split()[1]))

    plt.hist(energies, bins=args.num_bins)
    plt.savefig(args.out + "-energies.png")

    print(energies)

    plt.hist(rms, bins=args.num_bins)
    plt.savefig(args.out + "-rms.png")


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--num_bins', type=int, dest='num_bins', default=None, help='# histogram bins')
    parser.add_argument('--out', type=str, dest='out', default=None, help='File name for plot image')
    args = parser.parse_args()

    energy_plot(args)


if __name__ == '__main__':
    main()