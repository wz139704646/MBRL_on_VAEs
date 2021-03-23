import abc


class BaseConfiguration(metaclass=abc.ABCMeta):
    """base class for all kinds of configurations"""

    def __init__(self, **kwargs):
        # save the raw value for the serialization
        self.raw = kwargs

    @abc.abstractmethod
    def set_unnecessary_configs(self, **kwargs):
        """set configs that not necessarily provided by user"""
        pass

    @abc.abstractmethod
    def set_necessary_configs(self, **kwargs):
        """set configs that necessarily provided by user"""
        pass