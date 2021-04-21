import logging
import argparse

from load_config import load_config_from_files
from experiments.experiment_mbpo_vae import MBPOVAEsExperiment


def parse_args():
    """parse command line arguments"""
    desc = "MBPO VAE experiments (training and testing)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config-file', nargs='+', type=str,
                        help='experiemnt configuration file path(s) (yaml format supported now)')

    args = parser.parse_args()

    return args


def main(args):
    """main procedure"""
    try:
        cfgs = load_config_from_files(args.config_file)
        exp = MBPOVAEsExperiment(cfgs)

        exp.exec()
    except Exception as e:
        logging.error("MBPO VAE experiment encountered error: \n{}".format(e), exc_info=True)
    else:
        logging.info("MBPO VAE experiment done")


if __name__ == "__main__":
    args = parse_args()

    main(args)
