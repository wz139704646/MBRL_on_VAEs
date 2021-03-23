import yaml
from config import *


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

            return cfg_map
    except Exception as e:
        raise Exception("yaml config loading encountered error: {}".format(e))


if __name__ == '__main__':
    cfg = load_config_yaml("train_config_test.yml")
    print(cfg)
