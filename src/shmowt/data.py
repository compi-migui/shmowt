from pathlib import Path

import h5py
import pandas as pd


def load_raw_data(data_path):
    with h5py.File(data_path, 'r') as data_file:
        data_all = data_file['datacompleto']
        # data = pd.DataFrame(data_all).transpose()
        # TODO: Get rid of this workaround once we have cleaner data
        data = _truncate_data(data_all)
    # noinspection PyTypeChecker
    data.insert(0, 'class', [class_mapper(i) for i in range(data.shape[0])])
    return data


def _load_tiny(data_path, n=200):
    '''
    Meant for quick development only. Take only a tiny sample of the dataset.
    :param data_path:
    :param n:
    :return:
    '''
    data_all = load_raw_data(data_path)
    return data_all.sample(n=n, weights='class', random_state=0, ignore_index=True)


def _truncate_data(data_all):
    # Keep only the first 403 samples in each trial.
    # Works around the fact we're using a data set with too large trials.
    columns = 58008  # row leng
    L = 2417  # samples per sensor per trial
    keep = 403  # how many samples per sensor per trial to actually keep
    keep_columns = [i for i in range(columns) if i % L < keep]
    data = pd.DataFrame(data_all[keep_columns,]).transpose()
    return data


def class_mapper(n):
    # Our dataset doesn't have metadata for what class each experiment belongs to,
    # so work it out from the order they're provided in.
    classes = [0, 1, 2, 3, 4]
    if n < 2460:
        return classes[0] # healthy
    elif 2460 <= n < 2460 + 820:
        return classes[1]
    elif 2460 <= n < 2460 + 820*2:
        return classes[2]
    elif 2460 <= n < 2460 + 820*3:
        return classes[3]
    elif 2460 <= n < 2460 + 820*4:
        return classes[4]