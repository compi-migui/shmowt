import functools
import os
from pathlib import Path
import tempfile

from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


from shmowt.config import get_config
from shmowt.data import load_raw_data, _load_tiny


def accuracy(tn, fp, fn, tp):
    return (tp+tn)/(tp+fp+fn+tn)


def precision(tn, fp, fn, tp):
    return tp/(tp+fp)


def sensitivity(tn, fp, fn, tp):
    # aka recall
    return tp/(tp+fn)


def f1_score(ppv, tpr):
    return 2*(ppv*tpr)/(ppv+tpr)


def specificity(tn, fp, fn, tp):
    return tn/(tn+fp)


def metrics_from_confusion_matrix(conf_matrix):
    metrics = {"acc": 0, "ppv": 0, "tpr": 0, "f1": 0, "tnr": 0}
    for cls in range(len(conf_matrix)):
        total = sum(sum(conf_matrix))
        predicted_positives = sum(conf_matrix[:, cls])
        predicted_negatives = total - predicted_positives
        tp = conf_matrix[cls][cls]
        fp = predicted_positives - tp
        fn = sum(conf_matrix[cls]) - tp
        tn = predicted_negatives - fn
        metrics["acc"] += accuracy(tn, fp, fn, tp)
        metrics["ppv"] += precision(tn, fp, fn, tp)
        metrics["tpr"] += sensitivity(tn, fp, fn, tp)
        metrics["tnr"] += specificity(tn, fp, fn, tp)
    for metric in metrics:
        metrics[metric] = metrics[metric] / len(conf_matrix)
    metrics["f1"] = f1_score(metrics["ppv"], metrics["tpr"])

    return metrics



if __name__ == '__main__':
    config = get_config(os.getenv('SHMOWT_CONFIG'))

    cache_path = config.get('cache', 'path', fallback=None)
    if cache_path is None:
        cache_path = Path(tempfile.gettempdir()) / 'shmowt-cache'
        cache_path.mkdir(exist_ok=True)
    memory = Memory(cache_path, verbose=config.get('debug', 'verbosity_memory', fallback=0))

    tiny_data = config.getboolean('debug', 'tiny_data', fallback=False)
    data_path = config.get('data', 'path')
    if tiny_data:
        load_tiny_cache = memory.cache(_load_tiny, verbose=config.get('debug', 'verbosity_memory', fallback=0))
        data_raw = load_tiny_cache(data_path)
    else:
        data_raw = load_raw_data(data_path)

    # PCA(n_components=0.85, svd_solver='full')
    pipeline_knn = Pipeline(
        [
            ('column_scaling', StandardScaler()),
            ('dim_reduction', PCA(svd_solver='full')),
            ('classification', KNeighborsClassifier()), # TODO: could do passthrough here
        ],
        memory=memory,
        verbose=config.getboolean('debug', 'verbose_pipelines', fallback=False)
    )
    # pipeline_svm = Pipeline(
    #     [
    #         ('column_scaling', StandardScaler()),
    #         ('dim_reduction', PCA(svd_solver='full')),
    #         ('classification', SVC())
    #     ],
    #     memory=config.getboolean('debug', 'verbose_pipelines', fallback=False),
    # )
    #
    # param_grid = {
    #     "dim_reduction__n_components": [0.85, 0.90, 0.95],
    #     "classification__n_neighbors": 10 #np.logspace(-4, 4, 4),
    # }

    pipeline_knn.set_params(
        dim_reduction__n_components=8,
        classification__n_neighbors=5
    )

    pipeline = pipeline_knn
    eval_indicators = dict()
    split_metrics = []
    kfold_splits = 5
    kf = KFold(n_splits=kfold_splits)
    for train, test in kf.split(data_raw):
        pipeline.fit(data_raw.loc[train, data_raw.columns.drop("class")], data_raw.loc[train, 'class'])

        predicted = pipeline.predict(data_raw.loc[test, data_raw.columns.drop("class")])
        true = data_raw.loc[test, 'class']

        conf_matrix = confusion_matrix(true, predicted)
        split_metrics.append(metrics_from_confusion_matrix(conf_matrix))

    for metric in split_metrics[0]:
        # Evaluator indicators are averaged from the indicators of all splits
        eval_indicators[metric] = sum([x[metric] for x in split_metrics]) / len(split_metrics)
    print(eval_indicators)

