import sys
import logging
import logging.config
from torch.utils.tensorboard import SummaryWriter


def get_default_logger():
    """set default logger setting and return the root logger"""
    # basic setting
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # default handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # default formatter
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s-%(name)s  %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def configure_logger(logger_config):
    """set up logger config
    :return: a logger object
    """
    if logger_config.dict is not None:
        logging.config.dictConfig(logger_config.dict)
    elif logger_config.path:
        logging.config.fileConfig(logger_config.path)
    else:
        raise Exception("logger configuration not found")

    if logger_config.name:
        return logging.getLogger(logger_config.name)
    else:
        return logging.getLogger()


def configure_summary_writer(summary_writer_config):
    """set up tensorboard summary writer config
    :return: a summary writer object
    """
    return SummaryWriter(**summary_writer_config)


def adapt_configure_log(log_config):
    """configure the logging related configuration
    :return: a dict, mapping the name of logging components to the component object
    """
    if log_config is None:
        return {"logger": get_default_logger()}

    log_comp = {}
    if log_config.logger_config is not None:
        log_comp["logger"] = configure_logger(log_config.logger_config)
    else:
        log_comp["logger"] = get_default_logger()
    if log_config.summary_writer_config is not None:
        log_comp["summary_writer"] = configure_summary_writer(log_config.summary_writer_config)

    return log_comp
