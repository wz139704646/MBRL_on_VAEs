# execute experiment about all kinds of vaes training and testing
import os
import time
import torch
import matplotlib.pyplot as plt

from models.vaes import *
from base_experiment import BaseExperiment
from handlers.train_test import adapt_train, adapt_test
from handlers.save_model import get_extra_setting
from handlers.load_data import adapt_load_data
from handlers.configure_optimizer import adapt_configure_optimizer
from handlers.configure_log import adapt_configure_log
from utils.exp import plot_losses, set_seed
from utils.data import prepare_data_custom_tensor


class VAEsExperiment(BaseExperiment):
    """VAE training and testing (basic) experiment"""

    def apply_configs(self):
        """apply the configurations"""
        try:
            # environment setting
            if "general" in self.exp_configs:
                set_seed(self.exp_configs['general'].seed)

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

            log_config = self.exp_configs["log"] if "log" in self.exp_configs else None
            self.log_comp = adapt_configure_log(log_config)
        except Exception:
            raise

    def prepare_data(self):
        """dataset preparation"""
        kwargs = {
            'num_workers': 1,'pin_memory': True
        } if self.exp_configs["train"].cuda else {}
        model_config = self.exp_configs["model"]
        dataset_config = self.exp_configs["dataset"]

        return adapt_load_data(model_config.model_name, self.dataset_file,
                               dataset_config, **kwargs)

    def train(self, epoch):
        """training process"""
        train_args = self.exp_configs["train"]
        model_config = self.exp_configs["model"]
        train_loader = self.dataloaders[0]

        return adapt_train(model_config.model_name, self.model, self.optimizer,
                           train_loader, train_args, epoch, self.log_comp)

    def test(self, epoch):
        """testing model"""
        train_args = self.exp_configs["train"]
        test_args = self.exp_configs["test"]
        model_config = self.exp_configs["model"]
        test_loader = self.dataloaders[1]

        return adapt_test(model_config.model_name, self.model, test_loader,
                          train_args, test_args, epoch, self.log_comp)

    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        try:
            model_config = self.exp_configs["model"]
            train_config = self.exp_configs["train"]

            self.dataloaders = self.prepare_data()
            self.model = eval(model_config.model_name)(**model_config.model_args)
            self.optimizer = adapt_configure_optimizer(
                model_config.model_name, self.model, train_config.optimizer_config)

            # tensorboard logging
            if "summary_writer" in self.log_comp:
                writer = self.log_comp["summary_writer"]
                writer.add_graph(self.model, torch.zeros(1, *self.model.input_size))
        except Exception:
            raise

    def run(self, **kwargs):
        """run the main part of the experiment"""
        try:
            train_config = self.exp_configs["train"]
            writer = None

            if "summary_writer" in self.log_comp:
                writer = self.log_comp["summary_writer"]

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

                    # tensorboard logging
                    if writer is not None:
                        writer.add_scalar("[loss/train] " + k, avg_loss[k], epoch-1)

                test_loss = self.test(epoch)
                # tensorboard logging
                if writer is not None:
                    for k in test_loss.keys():
                        writer.add_scalar("[loss/test]" + k, test_loss[k], epoch-1)

                # updates after each epoch
                self.model.update_epoch()
        except Exception:
            raise

    def _save(self, extra=None):
        """save the model and configurations"""
        train_config = self.exp_configs["train"]
        save_config = train_config.save_config

        cfgs = {}
        for cfg_key in save_config.store_cfgs:
            cfgs[cfg_key] = self.exp_configs[cfg_key].raw

        torch.save({
            "exp_configs": cfgs,
            "model_state_dict": self.model.state_dict(),
            "extra": extra}, self.model_file)

    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        # save model, plot losses
        try:
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
            if "summary_writer" in self.log_comp:
                writer = self.log_comp["summary_writer"]
                writer.close()
        except Exception:
            raise
