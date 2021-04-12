from base_config import BaseConfiguration
from train_config import SaveModelConfiguration
from test_config import SaveResultsConfiguration


class RLEnvConfiguration(BaseConfiguration):
    """class stores the env for rl configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(RLEnvConfiguration, self).__init__(**kwargs)
        # default values
        self.num = 1
        self.gamma = 0.99
        self.max_episode_steps = 100

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set rl env configs that necessarily provided by user"""
        try:
            self.env_name = kwargs["env_name"]
        except Exception as e:
            raise Exception("necessary configs error in rl env configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set rl env configs that not necessarily provided by user"""
        if "num" in kwargs:
            self.num = kwargs["num"]
        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]
        if "max_episode_steps" in kwargs:
            self.max_episode_steps = kwargs["max_episode_steps"]


class RLConfiguration(BaseConfiguration):
    """class stores the rl experiment configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(RLConfiguration, self).__init__(**kwargs)
        # default values
        self.device = 'cpu'
        self.save_model_config = None
        self.save_result_config = None
        self.model_load_path = ''
        self.buffer_load_path = ''
        self.log_interval = 10
        self.eval_interval = 1
        self.save_interval = 5

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set rl configs that necessarily provided by user"""
        try:
            self.env = RLEnvConfiguration(**kwargs["env"])
            self.algos = kwargs['algos']
        except Exception as e:
            raise Exception("necessary configs error in train configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set rl configs that not necessarily provided by user"""
        if "device" in kwargs:
            self.device = kwargs["device"]
        if "save_model_config" in kwargs:
            self.save_model_config = SaveModelConfiguration(**kwargs["save_model_config"])
        if "save_result_config" in kwargs:
            self.save_result_config = SaveResultsConfiguration(**kwargs["save_result_config"])
        if "model_load_path" in kwargs:
            self.model_load_path = kwargs["model_load_path"]
        if "buffer_load_path" in kwargs:
            self.buffer_load_path = kwargs["buffer_load_path"]
        if "log_interval" in kwargs:
            self.log_interval = kwargs["log_interval"]
        if "eval_interval" in kwargs:
            self.eval_interval = kwargs["eval_interval"]
        if "save_interval" in kwargs:
            self.save_interval = kwargs["save_interval"]
