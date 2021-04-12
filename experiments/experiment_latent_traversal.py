# execute experiment about latent traversal for vaes
import os
import time
import copy
import torch
import random
import matplotlib.pyplot as plt

from base_experiment import BaseExperiment
from models.vaes import *
from utils.exp import set_seed
from handlers.load_model import adapt_load_model
from handlers.env_collect import collect_data
from handlers.latent_traversal import adapt_latent_traversal


class LatentTraversalExperiment(BaseExperiment):
    """Latent traversal experiment"""

    def apply_configs(self):
        """apply the configuration"""
        try:
            # environment setting
            if "general" in self.exp_configs:
                set_seed(self.exp_configs['general'].seed)

            # ensure results directory
            test_config = self.exp_configs['test']
            save_config = test_config.save_config
            if not os.path.exists(save_config.default_dir):
                os.makedirs(save_config.default_dir)

        except Exception:
            raise

    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        try:
            # load model
            test_config = self.exp_configs['test']
            if test_config.load_config is not None:
                model_path = test_config.load_config.path
            elif 'model' in self.exp_configs:
                model_config = self.exp_configs['model']
                model_path = os.path.join(model_config.default_dir, model_config.path)
            else:
                model_path = ''

            if model_path == '':
                raise Exception("no model path in the experiment config")

            load_res = adapt_load_model(model_path)
            self.model = load_res["model"]
            self.saved_configs = load_res["configs"]
        except Exception:
            raise

    def get_model_name(self):
        """return the name of the model class"""
        if 'model' in self.exp_configs:
            return self.exp_configs["model"].model_name
        elif 'model' in self.saved_configs:
            return self.saved_configs["model"].model_name

        return ''

    def sample_dataset(self, num, source):
        """sample from the dataset"""
        # dataset config from this experiment config
        if "dataset" in self.exp_configs:
            dataset_config_new = self.exp_configs["dataset"]
        else:
            dataset_config_new = None

        # dataset config from the saved model file
        if "dataset" in self.saved_configs:
            dataset_config_old = self.saved_configs["dataset"]
        else:
            dataset_config_old = None

        if dataset_config_new is None and dataset_config_old is None:
            raise Exception("dataset related config not found")

        if source == "file":
            model_file = ''
            if dataset_config_new is not None and dataset_config_new.path != '':
                model_file = os.path.join(
                    dataset_config_new.default_dir, dataset_config_new.path)
            elif dataset_config_old is not None and dataset_config_old.path != '':
                model_file = os.path.join(
                    dataset_config_old.default_dir, dataset_config_old.path)

            if model_file != '':
                # read data from file
                dataset = torch.load(model_file)
                if num < dataset.size(0):
                    samples = dataset[random.sample(range(0, dataset.size(0)), num)]
                else:
                    samples = dataset
            else:
                raise Exception("unable to sample from dataset file (missing settings)")

        if source == "collect":
            # collect from env
            collect_config = dataset_config_new.collect_config \
                if dataset_config_new is not None else dataset_config_old.collect_config
            cfg = copy.deepcopy(collect_config)
            # enlarge the sample space to ensure randomness when the env is complicated;
            # for custom env, the space is at least one traversal batch
            cfg.dataset_size = num * 10 if cfg.type != "custom" else 1
            dataset = collect_data(collect_config)
            if num < dataset.size(0):
                samples = dataset[random.sample(range(0, dataset.size(0)), num)]
            else:
                samples = dataset

        return samples

    def latent_traversal(self, **kwargs):
        """do latent traversal"""
        test_config = self.exp_configs["test"]
        test_args = test_config.test_args["latent_traversal"]
        input_source = test_args["input_source"]
        num = test_args["num"]

        device = torch.device("cpu")
        # generate the base codes
        self.model.eval()

        if input_source == "sample":
            # random sample from the latent space
            codes = self.model.sample_latent(num, device, **kwargs)
        elif input_source == "file":
            # sample from dataset file and generate codes
            samples = self.sample_dataset(num, "file")
            codes = self.model.reparameterize(*self.model.encode(samples)).to(device)
        elif input_source == "collect":
            # sample from observations of env and generate codes
            samples = self.sample_dataset(num, "collect")
            codes = self.model.reparameterize(*self.model.encode(samples)).to(device)
        else:
            raise Exception("unknown input source for latent traversal")

        model_name = self.get_model_name()
        run_args = test_args["run_args"] if "run_args" in test_args else {}
        adapt_latent_traversal(model_name, self.model, codes, test_config.save_config, **run_args)

    def run(self, **kwargs):
        """run the main part of the experiment"""
        try:
            self.latent_traversal(**kwargs)
        except Exception:
            raise

    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        pass
