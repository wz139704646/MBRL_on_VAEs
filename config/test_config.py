from base_config import BaseConfiguration


class LoadModelConfiguration(BaseConfiguration):
    """class stores the model loading configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(LoadModelConfiguration, self).__init__(**kwargs)
        # default values
        self.path = ''

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set model loading configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set model loading configs that not necessarily provided by user"""
        if "path" in kwargs:
            self.path = kwargs["path"]


class SaveResultsConfiguration(BaseConfiguration):
    """class stores the results saving configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(SaveResultsConfiguration, self).__init__(**kwargs)
        # default values
        self.default_dir = './results'
        self.tag = ''

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set results saving configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set results saving configs that not necessarily provided by user"""
        if "default_dir" in kwargs:
            self.default_dir = kwargs["default_dir"]
        if "tag" in kwargs:
            self.tag = kwargs["tag"]


class TestConfiguration(BaseConfiguration):
    """class stores the testing configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(TestConfiguration, self).__init__(**kwargs)
        # default values
        self.load_config = None
        self.save_config = None
        self.test_args = None
        # initialize the extra field to pass some data
        self.extra = {}

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set testing configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set testing configs that not necessarily provided by user"""
        if "load_config" in kwargs:
            self.load_config = LoadModelConfiguration(**kwargs["load_config"])
        if "save_config" in kwargs:
            self.save_config = SaveResultsConfiguration(**kwargs["save_config"])
        if "test_args" in kwargs:
            self.test_args = kwargs["test_args"]
