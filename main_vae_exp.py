import argparse

from load_config import load_config_yaml
from experiments.experiment_vaes import VAEsExperiment


def parse_args():
    """parse command line arguments"""
    desc = "VAE basic experiments (training and testing)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config-file', type=str,
                        help='experiemnt configuration file path (yaml format supported now)')

    args = parser.parse_args()

    return args


def main(args):
    """main procedure"""
    try:
        cfgs = load_config_yaml(args.config_file)
        exp = VAEsExperiment(cfgs)

        exp.exec()
    except Exception as e:
        print("VAE experiment encountered error: \n{}".format(e))
    else:
        print("VAE experiment done")


if __name__ == "__main__":
    args = parse_args()

    main(args)