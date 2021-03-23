# execute experiment about all kinds of vaes training and testing
import os
import time
import torch
from torch import optim
import matplotlib.pyplot as plt

from models.vaes import *
from base_experiment import BaseExperiment
from train_test import train_vae, test_vae
from env_collect import collect_data
from save_model import get_extra_setting
from utils.exp import plot_losses
from utils.data import prepare_data_custom_tensor


class VAEsExperiment(BaseExperiment):
    """VAE training and testing (basic) experiment"""

    def apply_configs(self):
        """apply the configurations"""
        try:
            # environment setting
            if "general" in self.exp_configs:
                torch.manual_seed(self.exp_configs['general'].seed)

            # ensure directories
            dataset_config = self.exp_configs["dataset"]
            train_config = self.exp_configs["train"]
            model_config = self.exp_configs["model"]
            test_config = self.exp_configs["test"]

            # ensure dataset directory
            dataset_file = dataset_config.default_dir
            if dataset_config.path != '':
                dataset_file = os.path.join(dataset_file, dataset_config.path)
            else:
                dataset_file = os.path.join(dataset_file, "{}.pt".format(int(time.time())))
            if not os.path.exists(os.path.dirname(dataset_file)):
                os.makedirs(os.path.dirname(dataset_file))
            self.dataset_file = dataset_file

            # ensure model directory
            if train_config.save_config or model_config.path:
                self.save_model = True
                model_file = train_config.save_config.default_dir
                if train_config.save_config.path:
                    model_file = os.path.join(model_file, train_config.save_config.path)
                elif model_config.path:
                    # refer to model config
                    model_file = os.path.join(model_config.default_dir, model_config.path)
                else:
                    # use timestamp as name
                    model_file = model_file if model_file != '' else model_config.default_dir
                    model_file = os.path.join(model_file, "{}.pth.tar".format(int(time.time())))
                if not os.path.exists(os.path.dirname(model_file)):
                    os.makedirs(os.path.dirname(model_file))
                self.model_file = model_file
            else:
                self.save_model = False

            # ensure test results directory
            if test_config.save_config:
                self.save_test_results = True
                results_dir = test_config.save_config.default_dir
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                self.results_dir = results_dir
            else:
                self.save_test_results = False

            # other settings
            device = torch.device("cuda" if train_config.cuda else "cpu")
            self.exp_configs["train"].extra["device"] = device
            self.exp_configs["test"].extra["device"] = device
        except Exception as e:
            raise Exception("applying experiment config encountered error: {}".format(e))

    def prepare_data(self):
        """dataset preparation"""
        kwargs = {
            'num_workers': 1,'pin_memory': True
        } if self.exp_configs["train"].cuda else {}
        dataset_config = self.exp_configs["dataset"]

        if os.path.exists(self.dataset_file):
            # load from disk
            return tuple(prepare_data_custom_tensor(dataset_config.load_config.batch_size,
                                                    data_file=self.dataset_file,
                                                    shuffle=dataset_config.load_config.shuffle,
                                                    div=dataset_config.load_config.division, **kwargs))
        elif dataset_config.type == "collect":
            # collect from env
            data = collect_data(dataset_config.collect_config)
            return tuple(prepare_data_custom_tensor(dataset_config.load_config.batch_size, data=data,
                                                    shuffle=dataset_config.load_config.shuffle,
                                                    div=dataset_config.load_config.division, **kwargs))
        else:
            raise Exception("unable to prepare data")

    def train(self, epoch):
        """training process"""
        train_args = self.exp_configs["train"]
        train_loader = self.dataloaders[0]

        return train_vae(self.model, self.optimizer, train_loader, train_args, epoch)

    def test(self, epoch):
        """testing model"""
        train_args = self.exp_configs["train"]
        test_args = self.exp_configs["test"]
        test_loader = self.dataloaders[1]

        return test_vae(self.model, test_loader, train_args, test_args, epoch)

    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        try:
            model_config = self.exp_configs["model"]
            train_config = self.exp_configs["train"]

            self.dataloaders = self.prepare_data()
            self.model = eval(model_config.model_name)(**model_config.model_args)
            self.optimizer = optim.Adam(self.model.parameters(), lr=train_config.optimizer_config.lr)
        except Exception as e:
            raise Exception("preparation before run encountered error: {}".format(e))

    def run(self, **kwargs):
        """run the main part of the experiment"""
        try:
            train_config = self.exp_configs["train"]

            epochs = train_config.epochs
            self.losses = {}
            for epoch in range(1, epochs+1):
                avg_loss = self.train(epoch)
                # store all kinds of losses
                for k in avg_loss.keys():
                    if k in self.losses:
                        self.losses[k].append(avg_loss[k])
                    else:
                        self.losses[k] = [avg_loss[k]]

                self.test(epoch)

                # updates after each epoch
                self.model.update_epoch()
        except Exception as e:
            raise Exception("running experiment encountered error: {}".format(e))

    def _save(self, extra=None):
        """save the model and configurations"""
        train_config = self.exp_configs["train"]
        save_config = train_config.save_config

        cfgs = {}
        if save_config.store_model_config:
            cfgs["model"] = self.exp_configs["model"].raw
        if save_config.store_general_config:
            cfgs["general"] = self.exp_configs["general"].raw
        if save_config.store_dataset_config:
            cfgs["dataset"] = self.exp_configs["dataset"].raw
        if save_config.store_train_config:
            cfgs["train"] = self.exp_configs["train"].raw

        torch.save({
            "exp_configs": cfgs,
            "model_state_dict": self.model.state_dict(),
            "extra": extra}, self.model_file)

    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        # save model, plot losses
        if self.save_model:
            model_config = self.exp_configs["model"]
            extra_setting = get_extra_setting(model_config.model_name, self.model)
            self._save(extra=extra_setting)
        if self.save_test_results:
            test_config = self.exp_configs["test"]
            filename = "loss.png"
            filename = test_config.save_config.tag + "-" + filename \
                if test_config.save_config.tag else filename

            plot_losses(self.losses, os.path.join(self.results_dir, filename))
