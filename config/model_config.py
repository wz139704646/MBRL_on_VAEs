from base_config import BaseConfiguration


class ModelConfiguration(BaseConfiguration):
    """class stores the model configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(ModelConfiguration, self).__init__(**kwargs)
        # default values
        self.default_dir = './checkpoints'
        self.path = ''

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set model configs that necessarily provided by user"""
        try:
            self.model_name = kwargs["model_name"]
            self.model_args = kwargs["model_args"]
        except Exception as e:
            raise Exception("necessary configs error in model configuration: {}".format(e))

    def set_unnecessary_configs(self, **kwargs):
        """set model configs that not necessarily provided by user"""
        if "path" in kwargs:
            self.path = kwargs["path"]
