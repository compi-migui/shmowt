from pathlib import Path

import h5py
import pandas as pd


class Data:
    def __init__(self, name, path=None, data_func=None, load_cached=True):
        """
        :param name: Name of the data blob. Should be unique in the data path.
        :param path: Path to save/load cached data from.
        :param data_func: Function to call to generate data. Usually a functools.partial instance
        :param load_cached: Whether to load cached data, if available, as opposed to generating it.
        """
        # TODO: Could use less memory if we didn't load the entire file contents into the class. Would also be
        # slower, probably.
        self.name = name
        full_path = Path(path) / (name + ".hdf5")
        self.full_path = full_path
        if full_path.exists() and load_cached:
            # Just take cached data if available
            # TODO: Validate file via hash/etc rather than just filename?
            self.data = self._load_cached(full_path, name)
        elif data_func:
            # Fall back to actually
            self.data = data_func()
        else:
            # we're getting raw input data, not intermediate data
            with h5py.File(path, 'r') as data_file:
                if len(data_file) != 1:
                    raise IndexError(f"expected one object in file {path}, found {len(data_file)}")
                object_name = list(data_file.keys())[0]
                data_all = data_file[object_name]
                data = pd.DataFrame(data_all).transpose()
                if object_name == 'datacompleto':
                    # Our original input data is transposed for some reason.
                    data = data.transpose()
                self.data = data

    @staticmethod
    def _load_cached(full_path, name):
        with h5py.File(full_path, 'r') as data_file:
            return pd.DataFrame(data_file[name])

    def save(self):
        with h5py.File(self.full_path, 'w') as data_file:
            # TODO
            raise
            #data_file.create_dataset(self.name, self.data.shape, )


def load_raw_data(data_path):
    with h5py.File(data_path, 'r') as data_file:
        data_all = data_file['datacompleto']
        data = pd.DataFrame(data_all).transpose()
    data.insert(0, 'class', [class_mapper(i) for i in range(data.shape[0])])


def save_to_cache(data, name, cache_path, overwrite=True):
    save_path = Path(cache_path) / name