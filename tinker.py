import argparse
import os


def tinker_md(args):
    """
    Generate MD simulation for molecules from SMILES string input
    :param args: Argparse arguments
    :return: None
    """
    for _, _, files in os.walk(args.folder):
        for f in files:
            print(f.find("."))
            os.system("obabel --gen3D -ismi " + os.path.join(args.folder, f) + " -osdf " + "-O " +
                      os.path.join(args.out, f[:f.find(".") + 1] + "sdf"))
            tmp = open(os.path.join(args.out, f[:f.find(".") + 1] + "sdf"), "r")
            contents = tmp.readlines()
            contents[0] = f[:f.find(".")] + "\n"
            tmp.close()
            tmp = open(os.path.join(args.out, f[:f.find(".") + 1] + "sdf"), "w")
            for i in range(len(contents)):
                tmp.write(contents[i])
            tmp.close()
            os.system("sdf2tinkerxyz < " + os.path.join(args.out, f[:f.find(".") + 1] + "sdf"))
            tmp = open(os.path.join(args.out, f[:f.find(".") + 1] + "key"), "w")
            tmp.write("parameters    " + args.param_path + "\n")
            tmp.write("integrator    " + args.integrator+ "\n")
            tmp.write("archive" + "\n")
            tmp.close()
            os.system("mv " + f[:f.find(".") + 1] + "xyz " + args.out)
            os.system("rm " + f[:f.find(".") + 1] + "key ")
            os.system("dynamic " + os.path.join(args.out, f[:f.find(".") + 1] + "xyz") + " -k " +
                      os.path.join(args.out, f[:f.find(".") + 1] + "key") + " " + str(args.num_steps) + " " +
                      str(args.time_step) + " " + str(args.save_step) + " " + str(args.ensemble) + " " + str(args.temp))


def main():
    """
    Parse arguments and run tinkerMD function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, dest='folder', default=None, help='Folder path containing relevant files')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Folder path containing output files')
    parser.add_argument('--param_path', type=str, dest='param_path',
                        default="/data/swansonk1/anaconda3/envs/my-rdkit-env/Tinker-FFE/tinker/params/mmff",
                        help='File path to Tinker parameters')
    parser.add_argument('--integrator', type=str, dest='integrator', default="verlet",
                        help='File path to Tinker parameters')
    parser.add_argument('--num_steps', type=int, dest='num_steps', default=100000, help='Number of MD steps')
    parser.add_argument('--time_step', type=float, dest='time_step', default=1.0, help='Time step in femtoseconds')
    parser.add_argument('--save_step', type=float, dest='save_step', default=0.1, help='Time btw saves in picoseconds')
    parser.add_argument('--ensemble', type=int, dest='ensemble', default=2, help='1=NVE, 2=NVT')
    parser.add_argument('--temp', type=int, dest='temp', default=298, help='Temperature in degrees Kelvin')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=False)
    tinker_md(args)


if __name__ == '__main__':
    main()
