""" Compute evaluation metrics. """
from conformation.evaluate import main, Args

if __name__ == '__main__':
    main(Args().parse_args())
