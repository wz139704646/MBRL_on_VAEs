# provide all kinds of training functions
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def train_vae(model, optimizer, train_loader, train_args, epoch, log_comp):
    """basic VAE training process"""
    logger = log_comp["logger"]

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
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\nloss: {}'.format(
                epoch, batch_idx * len(data), dataset_size,
                100. * batch_idx / len(train_loader),
                loss_dict
            ))

    avg_loss = {}
    for k in train_loss.keys():
        avg_loss[k] = train_loss[k] / dataset_size * batch_size
    logger.info('=====> Epoch: {} Average loss: {}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def train_factor_vae(model, optimizer, train_loader, train_args, epoch, log_comp):
    """factor VAE training process
    :param optimizer: a dict of optimizer (for vae and discriminator)
    :param train_loader: list or tuple of dataloader, length 2
    """
    logger = log_comp["logger"]
    writer = None
    if "summary_writer" in log_comp:
        writer = log_comp["summary_writer"]

    model.train()
    train_loss = {}
    avg_D_acc = 0. # accuracy of discriminator
    batch_size = 0 # initialize
    dataset_size = len(train_loader[0].dataset)
    batch_num = len(train_loader[0])

    # factor vae needs two optimizers
    optimizer_vae = optimizer['vae']
    optimizer_disc = optimizer['discriminator']
    # factot vae needs two train loaders
    train_loader = zip(*[enumerate(l) for l in train_loader])

    # initialize loss function args
    loss_args = {"dataset_size": dataset_size, "optim_part": ''}

    for (batch_idx, data1), (_, data2) in enumerate(train_loader):
        data1 = data1.to(train_args.extra['device'])
        data2 = data2.to(train_args.extra['device'])
        if batch_size == 0:
            batch_size = data1.size(0)

        # train vae
        optimizer_vae.zero_grad()
        res = model(data1.clone())
        loss_args["optim_part"] = "vae"
        loss_dict = model.loss_function(*res, data1, **loss_args)

        # accumulate all kinds of losses
        for k in loss_dict.keys():
            stored_key = "vae " + k
            if stored_key in train_loss:
                train_loss[stored_key] += loss_dict[k].item()
            else:
                train_loss[stored_key] = loss_dict[k].item()

        vae_loss = loss_dict["loss"] # final loss
        vae_loss.backward(retain_graph=True)
        optimizer_vae.step()

        # train discriminator
        optimizer_disc.zero_grad()
        Dz = res[2] # use the previous result of vae
        z_prime = model(data2.clone(), no_dec=True)
        Dz_pperm = model.disc_permute_z(z_prime)
        loss_args["optim_part"] = "discriminator"
        loss_dict = model.loss_function(Dz, Dz_pperm, **loss_args)

        # accumulate all kinds of losses
        for k in loss_dict.keys():
            stored_key = "discriminator " + k
            if stored_key in train_loss:
                train_loss[stored_key] += loss_dict[k].item()
            else:
                train_loss[stored_key] = loss_dict[k].item()

        disc_loss = loss_dict["loss"] # final loss
        disc_loss.backward()
        optimizer_disc.step()

        # record the information of discriminator
        soft_Dz = F.softmax(Dz, 1)[:, :1].detach()
        soft_Dz_pperm = F.softmax(Dz_pperm, 1)[:, :1].detach()
        D_acc = ((soft_Dz >= 0.5).sum() + (soft_Dz_pperm < 0.5).sum()).float()
        D_acc /= 2 * batch_size
        avg_D_acc += D_acc.item()

        # updates after each train step
        model.update_step()

        if batch_idx % train_args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\nloss: {}\n'
                        'discriminator accuracy: {}'.format(
                epoch, batch_idx * len(data1), dataset_size,
                100. * batch_idx / batch_num,
                loss_dict, D_acc.item()
            ))

    avg_loss = {}
    for k in train_loss.keys():
        avg_loss[k] = train_loss[k] / dataset_size * batch_size
    avg_D_acc = avg_D_acc / dataset_size * batch_size
    logger.info('=====> Epoch: {} Average loss: {}'.format(
        epoch, avg_loss
    ))
    logger.info('=====> Epoch: {} Discriminator accuracy: {}'.format(
        epoch, avg_D_acc
    ))

    # tensorboard extra logging
    if writer is not None:
        writer.add_scalar("[accuracy/train] discriminator", avg_D_acc, epoch-1)

    return avg_loss


def test_vae(model, test_loader, train_args, test_args, epoch, log_comp):
    """basic VAE testing process"""
    logger = log_comp["logger"]

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
        test_loss[k] = test_loss[k] / dataset_size * batch_size
    logger.info('=====> Test set loss: {}'.format(test_loss))

    return test_loss


def test_factor_vae(model, test_loader, train_args, test_args, epoch, log_comp):
    """basic VAE testing process"""
    logger = log_comp["logger"]
    writer = None
    if "summary_writer" in log_comp:
        writer = log_comp["summary_writer"]

    model.eval()
    test_loss = {}
    avg_D_acc = 0. # accuracy of discriminator
    batch_size = 0 # initialize
    dataset_size = len(test_loader[0].dataset)
    loss_args = {"dataset_size": dataset_size, "optim_part": ""}

    test_loader = zip(*[enumerate(l) for l in test_loader])

    with torch.no_grad():
        for (i, data1), (_, data2) in enumerate(test_loader):
            data1 = data1.to(train_args.extra["device"])
            data2 = data2.to(train_args.extra["device"])
            if batch_size == 0:
                batch_size = data1.size(0)

            # vae loss testing
            res = model(data1.clone())
            loss_args["optim_part"] = "vae"
            loss_dict = model.loss_function(*res, data1, **loss_args)

            for k in loss_dict.keys():
                stored_key = "vae " + k
                if stored_key in test_loss:
                    test_loss[stored_key] += loss_dict[k]
                else:
                    test_loss[stored_key] = loss_dict[k]

            # discriminator loss testing
            Dz = res[2]
            z_prime = model(data2.clone(), no_dec=True)
            Dz_pperm = model.disc_permute_z(z_prime)
            loss_args["optim_part"] = "discriminator"
            loss_dict = model.loss_function(Dz, Dz_pperm, **loss_args)

            for k in loss_dict.keys():
                stored_key = "discriminator " + k
                if stored_key in test_loss:
                    test_loss[stored_key] += loss_dict[k]
                else:
                    test_loss[stored_key] = loss_dict[k]

            # record the information of discriminator
            soft_Dz = F.softmax(Dz, 1)[:, :1].detach()
            soft_Dz_pperm = F.softmax(Dz_pperm, 1)[:, :1].detach()
            D_acc = ((soft_Dz >= 0.5).sum() + (soft_Dz_pperm < 0.5).sum()).float()
            D_acc /= 2 * batch_size
            avg_D_acc += D_acc.item()

            if test_args.save_config and i == 0:
                # reconstruction testing
                recon_batch = model.reconstruct(data1)
                n = min(data1.size(0), 4)
                comparison = torch.cat([data1[:n], recon_batch[:n]])
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
        test_loss[k] = test_loss[k] / dataset_size * batch_size
    avg_D_acc = avg_D_acc / dataset_size * batch_size

    logger.info('=====> Test set loss: {}'.format(test_loss))
    logger.info('=====> Test set discriminator accuracy: {}'.format(avg_D_acc))

    # tensorboard extra logging
    if writer is not None:
        writer.add_scalar("[accuracy/test] discriminator", avg_D_acc, epoch-1)

    return test_loss


train_handlers = {
    "VAE": train_vae,
    "ConVAE": train_vae,
    "BetaVAE": train_vae,
    "ConvBetaVAE": train_vae,
    "BetaTCVAE": train_vae,
    "ConvBetaTCVAE": train_vae,
    "FactorVAE": train_factor_vae,
    "ConvFactorVAE": train_factor_vae,
    "SparseVAE": train_vae,
    "ConvSparseVAE": train_vae,
    "JointVAE": train_vae,
    "ConvJointVAE": train_vae
}
test_handlers = {
    "VAE": test_vae,
    "ConVAE": test_vae,
    "BetaVAE": test_vae,
    "ConvBetaVAE": test_vae,
    "BetaTCVAE": test_vae,
    "ConvBetaTCVAE": test_vae,
    "FactorVAE": test_factor_vae,
    "ConvFactorVAE": test_factor_vae,
    "SparseVAE": test_vae,
    "ConvSparseVAE": test_vae,
    "JointVAE": test_vae,
    "ConvJointVAE": test_vae
}


def adapt_train(model_name, model, optimizer, train_loader, train_args, epoch, log_comp):
    """adapt train function according to model name"""
    if model_name in train_handlers:
        return train_handlers[model_name](model, optimizer, train_loader, train_args, epoch, log_comp)
    else:
        raise Exception("no train handler for the model {}".format(model_name))


def adapt_test(model_name, model, test_loader, train_args, test_args, epoch, log_comp):
    """adapt test function according to model name"""
    if model_name in test_handlers:
        return test_handlers[model_name](model, test_loader, train_args, test_args, epoch, log_comp)
    else:
        raise Exception("no test handler for the model {}".format(model_name))
