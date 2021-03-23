from base_config import BaseConfiguration


class GeneralConfiguration(BaseConfiguration):
    """class stores the general configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(GeneralConfiguration, self).__init__(**kwargs)
        # default values
        self.seed = 1 # random seed

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set general configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set general configs that not necessarily provided by user"""
        if "seed" in kwargs:
            self.seed = kwargs["seed"]
