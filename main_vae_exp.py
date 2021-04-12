import logging
import argparse

from load_config import load_config_from_files
from experiments.experiment_vaes import VAEsExperiment


def parse_args():
    """parse command line arguments"""
    desc = "VAE basic experiments (training and testing)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config-file', nargs='+', type=str,
                        help='experiemnt configuration file path(s) (yaml format supported now)')

    args = parser.parse_args()

    return args


def main(args):
    """main procedure"""
    try:
        cfgs = load_config_from_files(args.config_file)
        exp = VAEsExperiment(cfgs)

        exp.exec()
    except Exception as e:
        logging.error("VAE experiment encountered error: \n{}".format(e), exc_info=True)
    else:
        logging.info("VAE experiment done")


if __name__ == "__main__":
    args = parse_args()

    main(args)
