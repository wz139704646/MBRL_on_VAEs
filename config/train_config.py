from base_config import BaseConfiguration


class OptimizerConfiguration(BaseConfiguration):
    """class stores the optimizer configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(OptimizerConfiguration, self).__init__(**kwargs)
        # default values
        self.lr = 1e-3

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set optimizer configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set optimizer configs that not necessarily provided by user"""
        if "lr" in kwargs:
            self.lr = kwargs["lr"]


class SaveModelConfiguration(BaseConfiguration):
    """class stores the model saving configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(SaveModelConfiguration, self).__init__(**kwargs)
        # default values
        self.default_dir = './checkpoints'
        self.path = ''
        self.store_model_config = True
        self.store_general_config = True
        self.store_dataset_config = True
        self.store_train_config = True

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
        if "store_model_config" in kwargs:
            self.store_model_config = kwargs["store_model_config"]
        if "store_general_config" in kwargs:
            self.store_general_config = kwargs["store_general_config"]
        if "store_dataset_config" in kwargs:
            self.store_dataset_config = kwargs["store_dataset_config"]
        if "store_train_config" in kwargs:
            self.store_train_config = kwargs["store_train_config"]


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
