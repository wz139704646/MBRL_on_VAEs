import yaml

from config import *
from utils.general import integrate_dicts


def load_config_from_files(filepaths):
    """load config from several files (merge the key-value pairs)"""
    cfgs = []

    for fp in filepaths:
        cfgs.append(load_config_yaml(fp))

    return integrate_dicts(cfgs)


def load_config_yaml(filepath):
    """load yaml config and return the config object dict"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cfg = f.read()
            cfg = yaml.load(cfg, Loader=yaml.FullLoader)

            cfg_map = {}
            if 'general' in cfg:
                cfg_map['general'] = GeneralConfiguration(**cfg['general'])
            if 'dataset' in cfg:
                cfg_map['dataset'] = DatasetConfiguration(**cfg['dataset'])
            if 'model' in cfg:
                cfg_map['model'] = ModelConfiguration(**cfg['model'])
            if 'train' in cfg:
                cfg_map['train'] = TrainConfiguration(**cfg['train'])
            if 'test' in cfg:
                cfg_map['test'] = TestConfiguration(**cfg['test'])
            if 'log' in cfg:
                cfg_map['log'] = LogConfiguration(**cfg['log'])
            if 'rl' in cfg:
                cfg_map['rl'] = RLConfiguration(**cfg['rl'])

            return cfg_map
    except Exception as e:
        raise Exception("yaml config loading encountered error: {}".format(e))


if __name__ == '__main__':
    cfg = load_config_yaml("train_config_test.yml")
    print(cfg)
