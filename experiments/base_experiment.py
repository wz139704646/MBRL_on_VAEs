import abc


class BaseExperiment(metaclass=abc.ABCMeta):
    """base class for all kinds of experiments"""

    def __init__(self, exp_configs, hook_before_run=None, hook_after_run=None):
        """initialize the experiment
        :param exp_configs: the configurations needed in this experiment
        :param hook_before_run: hook function run before the main part run,
            take the experiment object itself as the only param
        :param hook_after_run: hook function run after the main part run,
            take the experiment object itself as the only param
        """
        self.exp_configs = exp_configs
        self.hook_before_run = hook_before_run
        self.hook_after_run = hook_after_run

    @abc.abstractmethod
    def apply_configs(self):
        """apply the configurations"""
        pass

    @abc.abstractmethod
    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        pass

    @abc.abstractmethod
    def run(self, **kwargs):
        """run the main part of the experiment"""
        pass

    @abc.abstractmethod
    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        pass

    def exec(self, **kwargs):
        """execute the entire experiment"""
        # apply the experiment configuration
        self.apply_configs()

        self.before_run(**kwargs)
        if self.hook_before_run is not None:
            self.hook_before_run(self)
        self.run(**kwargs)
        if self.hook_after_run is not None:
            self.hook_after_run(self)
        self.after_run(**kwargs)