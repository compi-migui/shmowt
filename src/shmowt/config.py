import configparser
from pathlib import Path


def get_config(config_path=None):
    if not config_path:
        config_path = Path.home() / 'shmowt' / 'contrib' / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    return config
