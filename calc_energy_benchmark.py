""" benchmarking """
from conformation.calc_energy_benchmark import calc_energy_benchmark, Args

if __name__ == '__main__':
    calc_energy_benchmark(Args().parse_args())
