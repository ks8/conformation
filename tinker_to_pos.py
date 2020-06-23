import numpy as np
import argparse
import os


def tinker_to_pos(args):
    """
    Convert tinker MD trajectories to conformation files
    :param args: Argparse arguments
    :return: None
    """
    for _, _, files in os.walk(args.input):
        for f in files:
            counter = 0
            if f[-3:] == "arc":
                with open(os.path.join(args.input, f), "r") as tmp:
                    line = tmp.readline()
                    pos = []
                    while line:
                        if line.split()[1] == f[:-4]:
                            if counter > 0:
                                pos = np.array(pos)
                                np.savetxt(os.path.join(args.out, "pos-" + str(counter - 1) + "-" + f[:-4] + ".txt"),
                                           pos)
                            pos = []
                            counter += 1
                        else:
                            pos.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                        line = tmp.readline()


def main():
    """
    Parse arguments and run conformers function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None, help='Folder name containing input files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder name for saving output')

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    tinker_to_pos(args)


if __name__ == '__main__':
    main()