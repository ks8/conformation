""" Compute evaluation metrics. """
from conformation.evaluate import evaluate, Args

if __name__ == '__main__':
    evaluate(Args().parse_args())
