import functools
import os
from pathlib import Path
import tempfile

from joblib import Memory
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


from shmowt.config import get_config
from shmowt.data import load_raw_data, _load_tiny

config = get_config(os.getenv('SHMOWT_CONFIG'))
cache_path = config.get('cache', 'path', fallback=None)
if cache_path is None:
    cache_path = Path(tempfile.gettempdir()) / 'shmowt-cache'
    cache_path.mkdir(exist_ok=True)
memory = Memory(cache_path, verbose=config.get('debug', 'verbosity_memory', fallback=0))


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


def noop_core(X):
    return X


def noop():
    """
    Used to trick sklearn into caching the final transformer in a pipeline.
    See https://github.com/scikit-learn/scikit-learn/issues/23112
    """
    return FunctionTransformer(noop_core)


@memory.cache
def reproduce_paper(data_raw, memory, verbose_pipelines=False):
    """
    This procedure applies column scaling and dimensionality reduction to the entire dataset, and only applies
    cross-validation to the classifier. Same as the paper does.
    :param data_raw:
    :param memory:
    :param verbose_pipelines:
    :return:
    """
    # Table 5 in the paper
    results = {"knn": [], "svm": []}
    kfold_splits = 5
    results["knn"] = pd.DataFrame(columns=['variance', 'pc_num', 'k', 'acc', 'ppv', 'tpr', 'f1', 'tnr'])
    results["svm"] = pd.DataFrame(columns=['variance', 'pc_num', 'ρ', 'acc', 'ppv', 'tpr', 'f1', 'tnr'])
    pipeline_overfit = Pipeline(
        [
            ('column_scaling', StandardScaler()),
            ('dim_reduction', PCA(svd_solver='full')),
            ('classification', noop())
        ],
        memory=memory,
        verbose=verbose_pipelines
    )
    pipeline_overfit.set_output(transform='pandas')

    explained_variances = [0.85, 0.90, 0.95]

    pipeline_knn_only = Pipeline(
        [
            ('classification', KNeighborsClassifier())
        ],
        memory=memory,
        verbose=verbose_pipelines
    )
    k = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 500]

    pipeline_svc_only = Pipeline(
        [
            ('classification', SVC(C=1.0, kernel='poly', degree=2, coef0=0))
        ],
        memory=memory,
        verbose=verbose_pipelines
    )
    kernel_scale = [5, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]

    for variance in explained_variances:
        pipeline_overfit.set_params(
            dim_reduction__n_components=variance
        )

        data = pipeline_overfit.fit_transform(data_raw.loc[:, data_raw.columns.drop("class")])
        # Adding back the class label makes the data consistent with the not-overfit case, avoiding special case
        # handling.
        data.insert(0, 'class', data_raw.loc[:, 'class'])
        if variance == 0.9:
            # No need to do this several times
            save_pca_scatter_plots(pca_data=data, prefix='reproduce')
        results_tmp = []
        for neighbors in k:
            if neighbors > data.shape[0]*(kfold_splits-1)/kfold_splits:
                # Can't use more k than there are training samples
                continue
            params = {'variance': variance, 'pc_num': pipeline_overfit.named_steps['dim_reduction'].n_components_,
                      'k': neighbors}
            pipeline_knn_only.set_params(
                classification__n_neighbors=neighbors
            )
            eval_indicators = cross_validation(data=data, pipeline=pipeline_knn_only, kfold_splits=kfold_splits)
            new_row = pd.Series(params | eval_indicators)
            # As recommended in official pandas docs https://pandas.pydata.org/docs/reference/api/pandas.concat.html
            # "It is not recommended to build DataFrames by adding single rows in a for loop.
            # Build a list of rows and make a DataFrame in a single concat."
            results_tmp.append(new_row)
        concat_me = [results["knn"]]
        concat_me.extend([new_row.to_frame().T for new_row in results_tmp])
        results["knn"] = pd.concat(concat_me, ignore_index=True)
        save_indicators_plot(results=results["knn"].loc[results["knn"]['variance'] == variance],
                             method="knn",
                             param_column='k',
                             prefix="reproduce")
        results_tmp = []
        for r in kernel_scale:
            params = {'variance': variance, 'pc_num': pipeline_overfit.named_steps['dim_reduction'].n_components_,
                      'ρ': r}
            pipeline_svc_only.set_params(
                classification__gamma=1/(r**2)
            )
            eval_indicators = cross_validation(data=data, pipeline=pipeline_svc_only, kfold_splits=kfold_splits)
            new_row = pd.Series(params | eval_indicators)
            results_tmp.append(new_row)
        concat_me = [results["svm"]]
        concat_me.extend([new_row.to_frame().T for new_row in results_tmp])
        results["svm"] = pd.concat(concat_me, ignore_index=True)
        save_indicators_plot(results=results["svm"].loc[results["svm"]['variance'] == variance],
                             method="svm",
                             param_column='ρ',
                             prefix="reproduce")
    return results


def save_pca_scatter_plots(pca_data, prefix=''):
    """
    Create and save scatter plots of PCA data
    :param pca_data: Array-like of data that already went through PCA
    :param prefix: string prepended to the saved figure filename
    :return:
    """
    for component_pair in [(0, 1), (0, 2), (0, 13), (0, 24)]:
        save_pca_scatter_plot(pca_data=pca_data, component_pair=component_pair, prefix=prefix)


def save_pca_scatter_plot(pca_data, component_pair, prefix=''):
    """
    Create a single scatter plot of Xth vs Yth principal component
    :param pca_data: Array-like of data that already went through PCA
    :param component_pair: tuple of two ints, the principal components to plot. 0 is the first principal component
    :param prefix: string prepended to the saved figure filename
    :return:
    """
    #
    fig, ax = plt.subplots(figsize=(6, 4),
                           layout='constrained',
                           frameon=False,  # No background color
                           )
    color_mapping = zip(['blue', 'purple', 'red', 'olive', 'lime'], [0, 1, 2, 3, 4])
    for color, dataset in color_mapping:
        ax.scatter(
            pca_data.loc[pca_data['class'] == dataset, f"pca{component_pair[0]}"],
            pca_data.loc[pca_data['class'] == dataset, f"pca{component_pair[1]}"],
            s=5,
            color=color,
            alpha=0.8,
            label=dataset
        )
    # plt.legend(loc="best", shadow=False, scatterpoints=1, markerscale=3)
    ax.set_xlabel(f"Principal component {component_pair[0]+1}")  # 0 is the 1st principal component and so on
    ax.set_ylabel(f"Principal component {component_pair[1]+1}")
    ax.grid(True)
    save_path = (Path(config.get('out', 'path'))
                 / f"{prefix}-pca-plot-{component_pair[0]+1}-vs-{component_pair[1]+1}.png")
    fig.savefig(save_path,
                format='png',
                transparent=False,
                dpi=200,
                bbox_inches='tight')


def save_indicators_plot(results, method, param_column, prefix=''):
    """
    Figures 10 and 11 in the Sensors paper
    """
    variance = results.iloc[0].at['variance']  # We assume the dataset was pre-selected
    fig, ax = plt.subplots(figsize=(6, 4),
                           layout='constrained',
                           frameon=False,  # No background color
                           )

    variants = results[param_column].unique()
    names = ["acc", "ppv", "tpr", "f1", "tnr"]
    for variant in variants:
        # ax.plot(names, results.loc[results[param_column] == variant, names], label=f"{param_column}={int(variant)}")
        # ax.plot(data=results.loc[results[param_column] == variant, names], label=f"{param_column}={int(variant)}")
        # Tansform dataframe so that pyplot is happy with its shape
        ax.plot(names, results.loc[results[param_column] == variant, names].iloc[0], label=f"{param_column}={int(variant)}")

    # TODO: add legend
    save_path = (Path(config.get('out', 'path'))
                 / f"{prefix}-indicators-plot-{method}-var{variance}.png")
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path,
                format='png',
                transparent=False,
                dpi=200,
                bbox_inches='tight')


def save_classifier_tables(results, prefix=''):
    """
    :param results: dict of DataFrames. keys are classifier names
    :param prefix: string prepended to the saved figure filename
    :return:
    """
    save_dir = Path(config.get('out', 'path'))

    for classifier in results:
        # 00% for explained variance, 0.0000% for all other floats (performance indicators)
        floatfmt = ['.0%', 'g', 'g']
        floatfmt.extend(['.2%'] * results[classifier].shape[1])
        save_path = save_dir / Path(f"{prefix}-results-table-{classifier}.md")
        results[classifier].to_markdown(buf=save_path,
                                        mode='wt',
                                        index=False,
                                        tablefmt='grid',
                                        floatfmt=floatfmt)
    pass


def cross_validation(data, pipeline, kfold_splits):
    eval_indicators = dict()
    split_metrics = []
    # Unlike the paper, this pipeline uses holdouts and cross-validation for the entire process, including scaling
    # and dimensionality reduction. Putting it aside for now for the sake of 100% reproduction of the paper's
    # results.
    # TODO: Come back to this later.
    kfold_splits = 5
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=0)
    for train, test in kf.split(data):
        pipeline.fit(data.loc[train, data.columns.drop("class")], data.loc[train, 'class'])
        # pipeline.fit(data_pca[train], data_raw.loc[train, 'class'])

        predicted = pipeline.predict(data.loc[test, data.columns.drop("class")])
        # predicted = pipeline.predict(data_pca[test])
        true = data.loc[test, 'class']

        conf_matrix = confusion_matrix(true, predicted)
        split_metrics.append(metrics_from_confusion_matrix(conf_matrix))

    for metric in split_metrics[0]:
        # Evaluator indicators are averaged from the indicators of all splits
        eval_indicators[metric] = sum([x[metric] for x in split_metrics]) / len(split_metrics)
    return eval_indicators


def main():

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


    pipeline_knn.set_params(
        dim_reduction__n_components=0.85,
        classification__n_neighbors=5
    )

    eval_indicators = reproduce_paper(data_raw, memory, config.getboolean('debug', 'verbose_pipelines', fallback=False))
    save_classifier_tables(eval_indicators, prefix='reproduce')
    # for method in eval_indicators:
    #     print('=======================================================')
    #     print(method)
    #     print(eval_indicators[method].to_string())
    #     print('=======================================================')
    pass



if __name__ == '__main__':
    main()
