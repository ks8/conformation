import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def energy_plot(args):
    """
    Plot histograms of energies and RMS for molecular conformations
    :param args: Argparse arguments
    :return: None
    """
    energies = []
    rms_list = []
    for _, _, files in os.walk(args.folder):
        for f in files:
            if f[:10] == "energy-rms":
                file = open(os.path.join(args.folder, f))
                contents = file.readlines()
                energy = float(contents[0].split()[1])
                rms = float(contents[1].split()[1])
                if args.lower_energy_bound < energy < args.upper_energy_bound:
                    energies.append(energy)
                    rms_list.append(rms)

    plt.hist(energies, bins=args.num_energy_bins)
    plt.title("Ethane ETKDG Energies")
    plt.ylabel("Frequency")
    plt.xlabel("Energy (kcal/mol)")
    plt.savefig(args.out + "-energies.png")

    plt.clf()
    plt.hist(rms_list, bins=args.num_rms_bins)
    plt.title("Ethane RMS values")
    plt.ylabel("Frequency")
    plt.xlabel("Distance ($\AA$)")
    plt.savefig(args.out + "-rms.png")


def main():
    """
    Parse arguments and execute file processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--num_energy_bins', type=int, dest='num_energy_bins', default=50,
                        help='# energy histogram bins')
    parser.add_argument('--num_rms_bins', type=int, dest='num_rms_bins', default=50, help='# rms histogram bins')
    parser.add_argument('--out', type=str, dest='out', default=None, help='File name for plot image')
    parser.add_argument('--upper_energy_bound', type=float, dest='upper_energy_bound', default=5.0,
                        help='Upper bound for molecule energy')
    parser.add_argument('--lower_energy_bound', type=float, dest='lower_energy_bound', default=-5.0,
                        help='Lower bound for molecule energy')
    args = parser.parse_args()

    energy_plot(args)


if __name__ == '__main__':
    main()
