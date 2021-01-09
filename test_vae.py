import os
import gym
import torch
import argparse
from torch import optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from utils.wrappers import make_atari, wrap_deepmind
from utils.data import prepare_data_custom_tensor
from models.vaes.vae import VAE


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "VAE reconstructs images observaed in atari game"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v0',
                        help='the name of atari game used as env (NoFrameskip required)'
                        ' (default "PongNoFrameskip-v0")')
    parser.add_argument('--dataset-size', type=int, default=64000, metavar='N',
                        help='whole dataset size (number of observations) (default 64000)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size for training (default 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between logging training status (default 10)')
    parser.add_argument('--n-hidden', type=int, default=400, metavar='N',
                        help='number of hidden units in MLP (default 400)')
    parser.add_argument('--dim-z', type=int, default=20, metavar='N',
                        help='dimension of latent space (default 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of optimizer (default 1e-3)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def collect_data(args):
    """collect data from observations"""
    env = wrap_deepmind(make_atari(args.env_name),
                        frame_stack=False, scale=True)
    obs = env.reset()  # initial observation
    data = []  # dataset buffer
    done = False
    for i in range(args.dataset_size):
        data.append(torch.from_numpy(
            obs).squeeze().unsqueeze(dim=0))  # add to buffer
        ac = env.action_space.sample()  # use random policy
        if not done:
            obs, _, done, _ = env.step(ac)
        else:
            obs = env.reset()
            done = False

        if (i+1) % 100 == 0:
            print("sampled {} frames".format(i+1))

    env.close()
    return torch.cat(data, dim=0)


def prepare_data(args, data_file, div=None, shuffle=True):
    """prepare data for training and testing
    :param div: a list of float, sumed up to 1, used to divide the dataset
    :return: a tuple of dataloader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if os.path.exists(data_file):
        # data file exists, load data from disk
        return tuple(prepare_data_custom_tensor(args.batch_size, data_file=data_file,
                                                shuffle=shuffle, div=div, **kwargs))
    else:
        # sample data from observations
        data = collect_data(args)
        return tuple(prepare_data_custom_tensor(args.batch_size, data_file=data_file,
                                                data=data, shuffle=shuffle, div=div, **kwargs))


def configuration(args):
    """set global configuration for initialization"""
    torch.manual_seed(args.seed)

    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['image_size'] = (84, 84)
    global_conf['image_channels'] = 1
    global_conf['data_dir'] = './dataset'
    global_conf['res_dir'] = './results'
    global_conf['data_division'] = [0.7, 0.3]  # train, test

    if not os.path.exists(global_conf['data_dir']):
        os.makedirs(global_conf['data_dir'])
    if not os.path.exists(global_conf['res_dir']):
        os.makedirs(global_conf['res_dir'])


def train(model, train_loader, epoch, optimizer, args, device, img_size):
    """VAE training process"""
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        decoded, encoded = model(data.view(-1, img_size[0]*img_size[1]))
        loss = model.loss_function(
            decoded, data.view(-1, img_size[0]*img_size[1]), encoded)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            ))

    avg_loss = train_loss / len(train_loader.dataset)
    print('=====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test(model, test_loader, epoch, args, device, img_size, res_dir):
    """VAE testing process"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            decoded, encoded = model(data.view(-1, img_size[0]*img_size[1]))
            test_loss += model.loss_function(
                decoded, data.view(-1, img_size[0]*img_size[1]), encoded).item()

            if i == 0:
                recon_batch = model.reconstruct(
                    data.view(-1, img_size[0]*img_size[1]))
                comparison = torch.cat([data[0], recon_batch.view(
                    args.batch_size, img_size[0], img_size[1])[0]])
                save_image(comparison.cpu(), res_dir +
                           '/reconstruction_'+str(epoch)+'.png')

    test_loss /= len(test_loader.dataset)
    print('=====> Test set loss: {:.4f}'.format(test_loss))


def main(args):
    """main procedure"""
    # get configuration
    device = global_conf["device"]
    img_size = global_conf["image_size"]
    data_dir = global_conf["data_dir"]
    res_dir = global_conf["res_dir"]
    div = global_conf['data_division']
    img_chs = global_conf['image_channels']

    # data file name
    data_file = os.path.join(data_dir, "{}_ch{}_{}x{}_{}.pt".format(
        args.env_name, img_chs, img_size[0], img_size[1], args.dataset_size))
    # prepare data
    train_loader, test_loader = prepare_data(args, data_file, div)

    # prepare model
    model = VAE(img_size[0]*img_size[1], args.n_hidden,
                args.dim_z, img_size[0]*img_size[1], binary=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train and test
    losses = []
    for epoch in range(1, args.epochs+1):
        avg_loss = train(model, train_loader, epoch,
                         optimizer, args, device, img_size)
        losses.append(avg_loss)
        test(model, test_loader, epoch, args, device, img_size, res_dir)
        with torch.no_grad():
            sample = model.sample(16, device).cpu()
            save_image(sample.view(
                16, 1, img_size[0], img_size[1]), res_dir+'/sample_'+str(epoch)+'.png', nrow=4)

    # plot train losses
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(res_dir+'/loss.png')


if __name__ == "__main__":
    args = parse_args()
    configuration(args)

    main(args)