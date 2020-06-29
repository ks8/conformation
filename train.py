from utils import loss_func
# noinspection PyUnresolvedReferences
from tqdm import tqdm


def train(model, optimizer, data, args, n_iter):
    """
    Function for training a normalizing flow model.
    :param n_iter: Number of training iterations completed so far
    :param model: nn.Module neural network
    :param optimizer: PyTorch optimizer
    :param data: DataLoader
    :param args: System args
    :return:
    """
    model.train()

    total_loss = 0
    loss_sum, iter_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        if args.cuda:
            batch = batch.cuda()
        model.zero_grad()
        z, log_jacobians = model(batch)
        loss = loss_func(z, log_jacobians, model.base_dist)
        loss_sum += loss.item()
        total_loss += loss_sum
        iter_count += len(batch)
        n_iter += len(batch)

        loss.backward()
        optimizer.step()

        if (n_iter // args.batch_size) % args.log_frequency == 0:
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0
            print("Loss avg = {:.4e}".format(loss_avg))

    print("Total loss = {:.4e}".format(total_loss))

    return n_iter, total_loss
