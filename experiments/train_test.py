# provide all kinds of training functions
import os
import torch
from torchvision.utils import save_image


def train_vae(model, optimizer, train_loader, train_args, epoch):
    """basic VAE training process"""
    model.train()
    train_loss = {}
    batch_size = 0 # initialize

    dataset_size = len(train_loader.dataset)
    loss_args = {"dataset_size": dataset_size}

    for batch_idx, data in enumerate(train_loader):
        data = data.to(train_args.extra['device'])
        if batch_size == 0:
            batch_size = data.size(0)

        optimizer.zero_grad()
        res = model(data.clone())
        loss_dict = model.loss_function(*res, data, **loss_args)
        print(loss_dict)

        # accumulate all kinds of losses
        for k in loss_dict.keys():
            if k in train_loss:
                train_loss[k] += loss_dict[k].item()
            else:
                train_loss[k] = loss_dict[k].item()

        loss = loss_dict["loss"] # final loss
        loss.backward()
        optimizer.step()

        # updates after each train step
        model.update_step()

        if batch_idx % train_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\nloss: {}'.format(
                epoch, batch_idx * len(data), dataset_size,
                100. * batch_idx / len(train_loader),
                loss_dict
            ))

    avg_loss = {}
    for k in train_loss.keys():
        avg_loss[k] = train_loss[k] / dataset_size * batch_size
    print('=====> Epoch: {} Average loss: {}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test_vae(model, test_loader, train_args, test_args, epoch):
    """basic VAE testing process"""
    model.eval()
    test_loss = {}
    batch_size = 0 # initialize

    dataset_size = len(test_loader.dataset)
    loss_args = {"dataset_size": dataset_size}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(train_args.extra["device"])
            if batch_size == 0:
                batch_size = data.size(0)

            # loss testing
            res = model(data.clone())
            loss_dict = model.loss_function(*res, data, **loss_args)

            for k in loss_dict.keys():
                if k in test_loss:
                    test_loss[k] += loss_dict[k]
                else:
                    test_loss[k] = loss_dict[k]

            if test_args.save_config and i == 0:
                # reconstruction testing
                recon_batch = model.reconstruct(data)
                n = min(data.size(0), 4)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                filename = "reconstruction_{}.png".format(epoch)
                filename = test_args.save_config.tag + "-" + filename \
                    if test_args.save_config.tag else filename
                save_image(
                    comparison.cpu(),
                    os.path.join(test_args.save_config.default_dir, filename),
                    nrow=4, pad_value=1)

        if test_args.save_config:
            # sampling testing
            sample = model.sample(16, test_args.extra["device"]).cpu()
            filename = "sample_{}.png".format(epoch)
            filename = test_args.save_config.tag + "-" + filename \
                if test_args.save_config.tag else filename
            save_image(
                sample,
                os.path.join(test_args.save_config.default_dir, filename),
                nrow=4, pad_value=1)

    for k in test_loss:
        test_loss[k] = test_loss[k] / len(test_loader.dataset) * batch_size
    print('=====> Test set loss: {}'.format(test_loss))

    return test_loss
