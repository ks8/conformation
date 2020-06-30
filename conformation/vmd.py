""" Visualize RDKit conformations in VMD. """
import argparse
from argparse import Namespace
import os


def vmd(args: Namespace) -> None:
    """
    Create files suitable for VMD input.
    :param args: Argparse arguments.
    :return: None.
    """
    f = open(os.path.join(args.input, "atoms.txt"))
    atoms = f.readlines()
    f.close()

    for _, _, files in os.walk(args.input):
        for f in files:
            if f[:3] == "pos":
                file = open(os.path.join(args.input, f))
                contents = file.readlines()
                out = open(os.path.join(args.out, f), "w")
                out.write(str(len(atoms)) + '\n')
                out.write('\n')
                for i in range(len(contents)):
                    out.write(atoms[i].split()[0] + "    ")
                    out.write(contents[i])


def main():
    """
    Parse arguments and execute file processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None, help='Folder path containing relevant files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Name of output folder')
    args = parser.parse_args()

    os.makedirs(args.out)
    vmd(args)


if __name__ == '__main__':
    main()
