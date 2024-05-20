import os
import functools

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from shmowt.config import get_config
from shmowt.data import Data


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


def insert_class_labels(data):
    data.insert(0, 'class', [class_mapper(i) for i in range(data.shape[0])])


def scale(data):
    # scaled value = (value - column_mean) / column_stdev
    scaler = StandardScaler()
    # We don't want to scale the class label, so skip that column
    data[data.columns.drop("class")] = scaler.fit_transform(data[data.columns.drop("class")])
    return data


def pca(components=8):
    pca = PCA(n_components=8)  # TODO: more components, comparing results for different numbers of components
    pca.fit(data[data.columns.drop("class")])  # excludes class label column


if __name__ == '__main__':
    config = get_config(os.getenv('SHMOWT_CONFIG'))
    data_path = config.get('data', 'path')
    cache_path = config.get('cache', 'path')

    data_raw = Data(name='raw', path=data_path)
    insert_class_labels(data_raw.data)
    data_scaled = Data(name='scaled', path=cache_path, data_func=functools.partial(scale, data_raw.data))
    data_scaled.save()
