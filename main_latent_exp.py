import logging
import argparse

from load_config import load_config_from_files
from experiments.experiment_latent_traversal import LatentTraversalExperiment


def parse_args():
    """parse command line arguments"""
    desc = "VAE latent space related experiments (training and testing)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config-file', nargs='+', type=str,
                        help='experiemnt configuration file path(s) (yaml format supported now)')

    args = parser.parse_args()

    return args


def main(args):
    """main procedure"""
    try:
        cfgs = load_config_from_files(args.config_file)
        exp = LatentTraversalExperiment(cfgs)

        exp.exec()
    except Exception as e:
        logging.error("Latent traversal experiment encountered error: \n{}".format(e), exc_info=True)
    else:
        logging.info("latent traversal experiment done")


if __name__ == "__main__":
    args = parse_args()

    main(args)