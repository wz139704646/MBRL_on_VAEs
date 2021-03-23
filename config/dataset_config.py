from base_config import BaseConfiguration


class CollectDatasetConfiguration(BaseConfiguration):
    """class that stores the arguments needed in env dataset collecting"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(CollectDatasetConfiguration, self).__init__(**kwargs)
        # default values
        self.type = "gym"
        self.custom_args = None

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set collect configs that necessarily provided by user"""
        try:
            self.dataset_size = kwargs["dataset_size"]
            self.env_name = kwargs["env_name"]
            # env related config (e.g. image frame size)
            self.env_config = kwargs["env_config"]
        except Exception as e:
            raise Exception("necessary configs error in collect configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set collect configs that not necessarily provided by user"""
        if "type" in kwargs:
            # dataset type
            self.type = kwargs["type"]
        if "custom_args" in kwargs:
            # custom env initialization args
            self.custom_args = kwargs["custom_args"]


class LoadDatasetConfiguration(BaseConfiguration):
    """class that stores dataset loading related configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(LoadDatasetConfiguration, self).__init__(**kwargs)
        # default values
        self.division = None
        self.shuffle = True

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set load configs that necessarily provided by user"""
        try:
            self.batch_size = kwargs["batch_size"]
        except Exception as e:
            raise Exception("necessary configs error in load configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set load configs that not necessarily provided by user"""
        if "division" in kwargs:
            # division ratio
            self.division = kwargs["division"]
        if "shuffle" in kwargs:
            # whether shuffle the dataset
            self.shuffle = kwargs["shuffle"]


class DatasetConfiguration(BaseConfiguration):
    """class stores the dataset related configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(DatasetConfiguration, self).__init__(**kwargs)
        # default values
        self.default_dir = './dataset' # default dataset directory
        self.path = ''
        self.collect_config = None
        self.load_config = None

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set dataset configs that necessarily provided by user"""
        try:
            self.type = kwargs["type"]
            if self.type == "existing":
                # existing dataset must accompany with path
                self.path = kwargs["path"]
        except Exception as e:
            raise Exception("necessary configs error in dataset configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set dataset configs that not necessarily provided by user"""
        if "default_dir" in kwargs:
            self.default_dir = kwargs["default_dir"]
        if "path" in kwargs:
            self.path = kwargs["path"]
        if "collect_config" in kwargs:
            self.collect_config = CollectDatasetConfiguration(**kwargs["collect_config"])
        if "load_config" in kwargs:
            self.load_config = LoadDatasetConfiguration(**kwargs["load_config"])
