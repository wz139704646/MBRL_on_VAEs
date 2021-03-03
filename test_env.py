import os
import math
import torch
import argparse
from torch import optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from envs import *
from models.vaes.beta_vae import ConvBetaVAE
from utils.model import save_
from utils.data import prepare_data_custom_tensor


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "ConvVAE reconstructs images observaed in custom environment"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env-name', type=str, default='GridWorld',
                        help='the name of custom env used ("GridWorld" supported now)'
                        ' (default "GridWorld")')
    parser.add_argument('--dataset-size', type=int, default=6400, metavar='N',
                        help='whole dataset size (number of observations) (default 8960)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between logging training status (default 10)')
    parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 64, 64], metavar='N',
                        help='number of channels of hidden layers (default [32 32 64 64])')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='number of hidden units in MLP layer before latent layer (default 256)')
    parser.add_argument('--dim-z', type=int, default=20, metavar='N',
                        help='dimension of latent space (default 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of optimizer (default 1e-3)')
    parser.add_argument('--beta', type=float, default=3., metavar=':math: `\\beta`',
                        help='beta coefficient of beta-VAE (default 3.)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the model (under the checkpoints path (defined in config))')
    parser.add_argument('--tag', type=str, default=None, metavar='T',
                        help='tag string for the save model file name (default None (no tag))')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def configuration(args):
    """set global configuration (including environment initialization args)"""
    torch.manual_seed(args.seed)

    global_conf['env_init_args'] = {
        'GridWorld': {
            'env_mode': 2,
            'no_wall': True,
            'no_door': True,
            'no_key': True}
    }
    global_conf['env_config'] = {
        'GridWorld': {'image_size': (3, 64, 64)}
    }

    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['data_dir'] = './dataset'
    global_conf['res_dir'] = os.path.join(
        './results/', "{}_ch{}".format(
            args.env_name, global_conf['env_config'][args.env_name]['image_size'][0]))
    global_conf['data_division'] = [0.7, 0.3]  # train, test
    global_conf['checkpoints_dir'] = os.path.join(os.path.dirname(__file__), './checkpoints')

    if not os.path.exists(global_conf['data_dir']):
        os.makedirs(global_conf['data_dir'])
    if not os.path.exists(global_conf['res_dir']):
        os.makedirs(global_conf['res_dir'])
    if not os.path.exists(global_conf['checkpoints_dir']):
        os.makedirs(global_conf['checkpoints_dir'])


def convert2tensor(ob):
    """convert ob data to tensor"""
    return torch.tensor(ob)


def get_ob_pytorch(env):
    """get the current ob data in tensor format with image shape in pytorch style"""
    origin_ob = env.get_ob()

    return convert2tensor(origin_ob.reshape(1, *origin_ob.shape))


def test_image(args, img_dir):
    """save initial image of the env for test"""
    # environment initialization
    env_init_args = global_conf["env_init_args"][args.env_name]
    env = eval(args.env_name)(**env_init_args)

    env.reset()
    origin_ob = env.get_ob()
    print("shape: {}, type: {}".format(origin_ob.shape, type(origin_ob)))

    ob = get_ob_pytorch(env)
    save_image(ob, img_dir+'/initial.png')

    traversal_obs = convert2tensor(env.traverse())
    save_image(traversal_obs, img_dir+'/traversal.png', nrow=8, pad_value=1)


def collect_data(args):
    """collect data from observation (traverse all the grid world)"""
    # initialize environment
    env_init_args = global_conf["env_init_args"][args.env_name]
    env = eval(args.env_name)(**env_init_args)

    # traverse and collect data
    traversal_obs = convert2tensor(env.traverse())
    data = []
    repeat_num = math.ceil(args.dataset_size / traversal_obs.size(0))

    for _ in range(repeat_num):
        data.append(traversal_obs.clone().detach())

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
        print("data shape: {}".format(data.size()))
        return tuple(prepare_data_custom_tensor(args.batch_size, data_file=data_file,
                                                data=data, shuffle=shuffle, div=div, **kwargs))


def train(model, train_loader, epoch, optimizer, args, device):
    """VAE training process"""
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        decoded, encoded = model(data.clone())
        loss = model.loss_function(decoded, data, encoded)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

    avg_loss = train_loss / len(train_loader.dataset) * args.batch_size
    print('=====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test(model, test_loader, epoch, args, device, res_dir):
    """VAE testing process"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            decoded, encoded = model(data.clone())
            test_loss += model.loss_function(decoded, data, encoded).item()

            if i == 0:
                recon_batch = model.reconstruct(data)
                n = min(data.size(0), 2)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(), res_dir +
                           '/reconstruction_'+str(epoch)+'.png', nrow=n, pad_value=1)

    test_loss = test_loss / len(test_loader.dataset) * args.batch_size
    print('=====> Test set loss: {:.4f}'.format(test_loss))


def main(args):
    """main procedure"""
    # get configuration
    device = global_conf["device"]
    img_size = global_conf["env_config"][args.env_name]['image_size']
    data_dir = global_conf["data_dir"]
    res_dir = global_conf["res_dir"]
    div = global_conf['data_division']
    save_dir = global_conf["checkpoints_dir"]

    # data file name
    data_file = os.path.join(data_dir, "{}_ch{}_{}x{}_{}.pt".format(
        args.env_name, img_size[0], img_size[1], img_size[2], args.dataset_size))
    # prepare data
    train_loader, test_loader = prepare_data(args, data_file, div)

    # prepare model
    model = ConvBetaVAE(img_size, args.channels, args.hidden_size, args.dim_z, args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train and test
    losses = []
    for epoch in range(1, args.epochs+1):
        avg_loss = train(model, train_loader, epoch,
                         optimizer, args, device)
        losses.append(avg_loss)
        test(model, test_loader, epoch, args, device, res_dir)
        with torch.no_grad():
            sample = model.sample(16, device).cpu()
            save_image(sample, res_dir+'/sample_'+str(epoch)+'.png', nrow=4, pad_value=1)

    # plot train losses
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(res_dir+'/loss.png')

    # save the model and related params
    if args.save:
        save_dir = os.path.join(save_dir, '{}_ch{}'.format(args.env_name, img_size[0]))
        save_(model, save_dir, args, global_conf, comment=args.tag)


if __name__ == "__main__":
    args = parse_args()
    configuration(args)

    # img_dir = os.path.join('./results/', "env_test")
    # test_image(args, img_dir)

    main(args)