from base_config import BaseConfiguration


class LoggerConfiguration(BaseConfiguration):
    """class stores the logging logger (normal logs) configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(LoggerConfiguration, self).__init__(**kwargs)
        # default values
        self.path = ''
        self.dict = None
        self.name = ''

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set logging logger configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set logging logger configs that not necessarily provided by user"""
        if "path" in kwargs:
            self.path = kwargs["path"]
        if "dict" in kwargs:
            self.dict = kwargs["dict"]
        if "name" in kwargs:
            self.name = kwargs["name"]


class LogConfiguration(BaseConfiguration):
    """class stores the logging related configuration"""

    def __init__(self, **kwargs):
        """initialize settings"""
        super(LogConfiguration, self).__init__(**kwargs)
        # default values
        self.logger_config = None
        self.summary_writer_config = None

        self.set_necessary_configs(**kwargs)
        self.set_unnecessary_configs(**kwargs)

    def set_necessary_configs(self, **kwargs):
        """set general configs that necessarily provided by user"""
        pass

    def set_unnecessary_configs(self, **kwargs):
        """set general configs that not necessarily provided by user"""
        if "logger_config" in kwargs:
            self.logger_config = LoggerConfiguration(**kwargs["logger_config"])
        if "summary_writer_config" in kwargs:
            self.summary_writer_config = kwargs["summary_writer_config"]
