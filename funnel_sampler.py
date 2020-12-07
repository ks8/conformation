""" Samples from Neal's funnel distribution """
from conformation.funnel_sampler import funnel_sampler, Args

if __name__ == '__main__':
    funnel_sampler(Args().parse_args())
