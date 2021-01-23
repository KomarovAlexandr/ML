import pandas, seaborn, numpy
import matplotlib.pyplot as plt
import additional_func as af
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn import metrics

import warnings
warnings.simplefilter('ignore')

pandas.set_option('display.max_columns', None)
DataFrame = pandas.read_csv('train.csv')

# Получение таблицы с признаками, их типом, максимальным и минимальным значениями
DataFrame.describe().to_excel('data_describe.xlsx')

DataFrame = DataFrame[(DataFrame[["Id"]] % 2 == 0).all(axis=1)]

# Удаление столбцов с большим кол-вом пустых значений
DataFrame = DataFrame.drop(['PoolQC', 'Alley', 'FireplaceQu', 'Fence', 'MiscFeature', 'LotFrontage'], axis=1)
# Фильтрация выбросов
DataFrame = DataFrame[(DataFrame[["GrLivArea"]] < 4000).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["GarageCars"]] < 4).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["OverallQual"]] > 2).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["1stFlrSF"]] < 2000).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["GrLivArea"]] < 2500).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["TotalBsmtSF"]] < 2000).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["TotalBsmtSF"]] > 450).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["GarageArea"]] < 1000).all(axis=1)]

# Добавление признака с ценой за один квадратный фут жилой площади
DataFrame.loc[:, 'PricePerFoot'] = DataFrame['SalePrice'] / DataFrame['GrLivArea']

# Удаление нулей и обновление индексов
DataFrame = DataFrame.dropna()
DataFrame.reset_index(drop=True)

for feature in DataFrame.columns:
    if DataFrame[feature].dtype == object:
        DataFrame[feature] = pandas.Categorical(DataFrame[feature])
        sex_map_train = dict(zip(DataFrame[feature].cat.codes, DataFrame[feature]))
        DataFrame[feature] = DataFrame[feature].cat.codes

# Вывод значений корреляции всех признаков с ценой в табличку
corr = DataFrame[['SalePrice'] + DataFrame.columns.to_list()].corr().iloc[0]
corr.sort_values().to_excel('correlation.xlsx')

# Разбиение цены на категориальный признак
labels = [1, 2, 3]
price_group = pandas.cut(DataFrame['SalePrice'],
                    bins=[DataFrame['SalePrice'].min(),
                          145000,
                          200000,
                          DataFrame['SalePrice'].max()],
                    labels=labels)
DataFrame.loc[:, 'price_group'] = numpy.array(price_group)

# Дополнительное удаление пустых строк. Почему то это нужно сделать еще раз, иначе бо-бо
DataFrame = DataFrame.dropna()

# ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
attributes = ['PricePerFoot', 'OverallQual', 'GrLivArea', 'YearBuilt', 'SalePrice']
# attributes = ['price_group', 'OverallQual']
# attributes = ['PricePerFoot', 'OverallQual']
# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = DataFrame[attributes]

# Построение карты корреляций
plt.figure()
corr = DataFrame[attributes + ['SalePrice']].corr()
sns_hmap = seaborn.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")
corr = DataFrame[attributes + ['SalePrice']].corr().iloc[0]
corr.sort_values().to_excel('corr.xlsx')

# таблицы для записи итоговых результатов
result = pandas.DataFrame()
time_result = pandas.DataFrame()

# ОБУЧЕНИЕ МОДЕЛЕЙ
# Оригинальные данные
af.scatter_plot_func(DataFrame, data_num, 'price_group', "ORIGINAL")
# Стандартизация
df_stand = preprocessing.scale(data_num)
df_stand = pandas.DataFrame(data=df_stand, index=data_num.index, columns=attributes)
af.scatter_plot_func(DataFrame, df_stand, 'price_group', "STANDARDIZATION")
# Нормализация
df_norm = PCA(n_components=2).fit_transform(data_num)
df_norm = preprocessing.normalize(data_num)
df_norm = pandas.DataFrame(data=df_norm, index=data_num.index, columns=attributes)
af.scatter_plot_func(DataFrame, df_norm, 'price_group', "NORMALIZATION")

"""
data = df_norm.to_numpy()
# data.loc[:, 'price_group'] = data_num['price_group']

plt.figure()
db = DBSCAN(eps=0.01, min_samples=10).fit(data)
core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("data size = ", len(data_num))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in numpy.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    print('xy = ', len(xy))
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
"""
"""
from scipy.cluster.hierarchy import linkage, dendrogram
# Извлекаем измерения как массив NumPy
samples = df_norm.values

# Реализация иерархической кластеризации при помощи функции linkage
mergings = linkage(samples, method='complete')

#varieties = list(df_norm.pop('price_group'))

# Строим дендрограмму, указав параметры удобные для отображения
dendrogram(mergings,
           #labels=varieties,
           leaf_rotation=90,
           leaf_font_size=1,
           )
"""

reduced_data = PCA(n_components=2).fit_transform(data_num)

plt.figure()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

# reduced_data = df_stand.to_numpy()
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
kmeans.fit(reduced_data)

labels = kmeans.labels_
print("silhouette_score ", metrics.silhouette_score(reduced_data, labels, metric='euclidean'))
print("davies_bouldin_score ", metrics.davies_bouldin_score(reduced_data, labels))

h = 10
# Граничные значения и значения сетки
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))
# Получим результат для каждой точки сетки и выведем диаграмму
Z = kmeans.predict(numpy.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Построим центроиды (центры кластеров) на диаграмме в виде крестиков
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

#
# # обновление индексации таблиц для корректного отображения
# result = result.sort_values(['it']).set_index(['it', 'type'], drop=True)
# time_result = time_result.sort_values(['it']).set_index(['it', 'type'], drop=True)
#
# # Вывод в таблицы excel
# result.to_excel('result.xlsx')
# time_result.to_excel('time_result.xlsx')
#
# print(result)
# print(time_result)

plt.show()
pandas.reset_option('max_columns')

