from base_config import BaseConfiguration
from model_config import ModelConfiguration
from train_config import SaveModelConfiguration, TrainConfiguration
from test_config import SaveResultsConfiguration, TestConfiguration


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


class RLEncodingConfiguration(BaseConfiguration):
    """class stores the rl encoding configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(RLEncodingConfiguration, self).__init__(**kwargs)
        # default values
        self.train_config = None
        self.test_config = None
        self.division = None
        self.max_update_steps = float("inf")
        self.max_test_steps = 30

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set encoding configs that necessarily provided by user"""
        try:
            self.model_config = ModelConfiguration(**kwargs["model_config"])
            self.batch_size = kwargs["batch_size"]
        except Exception as e:
            raise Exception("necessary configs error in rl encoding configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set encoding configs that unnecessarily provided by user"""
        if "train_config" in kwargs:
            self.train_config = TrainConfiguration(**kwargs["train_config"])
        if "test_config" in kwargs:
            self.test_config = TestConfiguration(**kwargs["test_config"])
        if "division" in kwargs:
            self.division = kwargs["division"]
        if "max_update_steps" in kwargs:
            self.max_update_steps = kwargs["max_update_steps"]
        if "max_test_steps" in kwargs:
            self.max_test_steps = kwargs["max_test_steps"]


class RLConfiguration(BaseConfiguration):
    """class stores the rl experiment configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(RLConfiguration, self).__init__(**kwargs)
        # default values
        self.device = 'cpu'
        self.save_model_config = None
        self.save_result_config = None
        self.encoding_config = None
        self.model_load_path = ''
        self.buffer_load_path = ''
        self.encoding_load_path = ''
        self.extra_load_path = ''
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
        if "encoding_config" in kwargs:
            self.encoding_config = RLEncodingConfiguration(**kwargs["encoding_config"])
        if "model_load_path" in kwargs:
            self.model_load_path = kwargs["model_load_path"]
        if "buffer_load_path" in kwargs:
            self.buffer_load_path = kwargs["buffer_load_path"]
        if "encoding_load_path" in kwargs:
            self.encoding_load_path = kwargs["encoding_load_path"]
        if "extra_load_path" in kwargs:
            self.extra_load_path = kwargs["extra_load_path"]
        if "log_interval" in kwargs:
            self.log_interval = kwargs["log_interval"]
        if "eval_interval" in kwargs:
            self.eval_interval = kwargs["eval_interval"]
        if "save_interval" in kwargs:
            self.save_interval = kwargs["save_interval"]
