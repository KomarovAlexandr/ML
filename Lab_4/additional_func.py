import matplotlib.pyplot as plt
import numpy, pandas
from matplotlib import cm
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

"""
Построение гистограммы распределения по входному признаку датафрейма
"""
def build_bar_chart(df, target, name='', change_x_ticks=False, angle=30):
    plt.figure(num=name + target)
    plt.suptitle(name + target)
    counts = df[target].value_counts()
    plt.bar(counts.index, counts.values)
    x_ticks_mas = []
    if change_x_ticks:
        for i in numpy.arange(0, 1, 0.1):
            x_ticks_mas.append(counts.index[int(len(counts.index) * i)])
            plt.xticks(x_ticks_mas)
    plt.xticks(rotation=angle)


"""
Функция построения диаграммы рассеяния
"""
def scatter_plot_func(df, data_num, target, name):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data_num)
    fig = plt.figure()
    fig.suptitle(name)
    labels = df[target].unique()
    plots = []
    colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))
    for ing, ng in enumerate(labels):
        plots.append(plt.scatter(x=data[df[target] == ng, 0],
                                 y=data[df[target] == ng, 1],
                                 c=colors[ing],
                                 edgecolor='k'))
    plt.xlabel("component1")
    plt.ylabel("component2")
    plt.legend(plots, labels, loc="lower right", title="species")


"""
Визуализация данных
"""
def show_klasters(reduced_data, model, labels_true, method):
    print("-------", method, "--------")
    labels = model.fit_predict(reduced_data)
    print("silhouette_score ", metrics.silhouette_score(reduced_data, labels, metric='euclidean'))
    print("davies_bouldin_score ", metrics.davies_bouldin_score(reduced_data, labels))
    print("adjusted_rand_score", metrics.adjusted_rand_score(labels_true, labels))

    h_x = (reduced_data[:, 0].max() - reduced_data[:, 0].min()) / 200
    h_y = (reduced_data[:, 1].max() - reduced_data[:, 1].min()) / 200
    # Граничные значения и значения сетки
    x_min, x_max = reduced_data[:, 0].min() - h_x, reduced_data[:, 0].max() + h_x
    y_min, y_max = reduced_data[:, 1].min() - h_y, reduced_data[:, 1].max() + h_y
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h_x), numpy.arange(y_min, y_max, h_y))

    # Получим результат для каждой точки сетки и выведем диаграмму
    try:
        Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Построим центроиды (центры кластеров) на диаграмме в виде крестиков
        try:
            centroids = model.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=169, linewidths=3,
                        color='w', zorder=10)
        except:
            print("no centroids for ", method)

        plt.title(method)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
    except:
        fig = plt.figure()
        # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], )
        fig.suptitle(method)
        plt.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], c=labels)
        # labels_unique = numpy.unique(labels, return_counts=False)
        # labels = labels.unique()
        # plots = []
        # colors = cm.rainbow(numpy.linspace(0, 1, len(labels_unique)))
        # for ing, ng in enumerate(labels_unique):
        #     plots.append(plt.scatter(x=reduced_data[ng, 0],
        #                              y=reduced_data[ng, 1],
        #                              c=colors[ing],
        #                              edgecolor='k'))
        # plt.xlabel("component1")
        # plt.ylabel("component2")
        # plt.legend(plots, labels, loc="lower right", title="species")




