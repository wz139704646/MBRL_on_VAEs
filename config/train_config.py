from base_config import BaseConfiguration


class OptimizerConfiguration(BaseConfiguration):
    """class stores the optimizer configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(OptimizerConfiguration, self).__init__(**kwargs)
        # default values
        self.lr = 1e-3
        self.subs = {} # multiple optimizer nodes

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set optimizer configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set optimizer configs that not necessarily provided by user"""
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        if "subs" in kwargs:
            subs_cfg = kwargs["subs"]
            for name in subs_cfg.keys():
                self.subs[name] = OptimizerConfiguration(**subs_cfg[name])


class SaveModelConfiguration(BaseConfiguration):
    """class stores the model saving configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(SaveModelConfiguration, self).__init__(**kwargs)
        # default values
        self.default_dir = './checkpoints'
        self.path = ''
        self.store_cfgs = []

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set model saving configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set model saving configs that not necessarily provided by user"""
        if "default_dir" in kwargs:
            self.default_dir = kwargs["default_dir"]
        if "path" in kwargs:
            self.path = kwargs["path"]
        if "store_cfgs" in kwargs:
            self.store_cfgs = kwargs['store_cfgs']


class TrainConfiguration(BaseConfiguration):
    """class stores the training configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(TrainConfiguration, self).__init__(**kwargs)
        # default values
        self.cuda = False
        self.log_interval = 10
        self.optimizer_config = OptimizerConfiguration()
        self.save_config = None
        # initialize the extra field to pass some data
        self.extra = {}

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set train configs that necessarily provided by user"""
        try:
            self.epochs = kwargs["epochs"]
        except Exception as e:
            raise Exception("necessary configs error in train configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set train configs that not necessarily provided by user"""
        if "cuda" in kwargs:
            self.cuda = kwargs["cuda"]
        if "log_interval" in kwargs:
            self.log_interval = kwargs["log_interval"]
        if "optimizer_config" in kwargs:
            self.optimizer_config = OptimizerConfiguration(**kwargs["optimizer_config"])
        if "save_config" in kwargs:
            self.save_config = SaveModelConfiguration(**kwargs["save_config"])
