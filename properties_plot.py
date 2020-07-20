""" Plot distributions of conformation properties. """
from conformation.properties_plot import properties_plot, Args

if __name__ == '__main__':
    args = Args().parse_args()
    properties_plot(args)
