import pandas, seaborn, numpy
import matplotlib.pyplot as plt
import additional_func as af
from sklearn import preprocessing
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.simplefilter('ignore')

pandas.set_option('display.max_columns', None)
DataFrame = pandas.read_csv('train.csv')

# Получение таблицы с признаками, их типом, максимальным и минимальным значениями
DataFrame.describe().to_excel('data_describe.xlsx')
print(DataFrame.describe())

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

# Удаление нулей и обновление индексов
DataFrame = DataFrame.dropna()
DataFrame.reset_index(drop=True)

print("Dataframe size = ", len(DataFrame))

for feature in DataFrame.columns:
    if DataFrame[feature].dtype == object:
        DataFrame[feature] = pandas.Categorical(DataFrame[feature])
        sex_map_train = dict(zip(DataFrame[feature].cat.codes, DataFrame[feature]))
        DataFrame[feature] = DataFrame[feature].cat.codes

# Вывод значений корреляции всех признаков с ценой в табличку
corr = DataFrame[['SalePrice'] + DataFrame.columns.to_list()].corr().iloc[0]
corr.sort_values().to_excel('correlation.xlsx')

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
# attributes = ['GarageCars', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageArea', '1stFlrSF',
#               'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
#               'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'HeatingQC', 'GarageType']

DataFrame.loc[:, 'PricePerFoot'] = DataFrame['SalePrice'] / DataFrame['GrLivArea']

attributes = ['PricePerFoot', 'OverallQual', 'GrLivArea', 'YearBuilt', 'SalePrice']
DataFrame.plot(kind="scatter", x="YearBuilt", y="PricePerFoot")
print(DataFrame.head())


# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = DataFrame[attributes]
# ЦЕЛЕВОЙ ПРИЗНАК
df_target = DataFrame['SalePrice']

data_num.hist(bins=100)
scatter_matrix(data_num, figsize=(20,6))

# Отрисовка гистограмм всех признаков из итогового набора
# for i in attributes:
#     af.build_bar_chart(data_num, i)

# Построение карты корреляций
plt.figure()
corr = DataFrame[attributes + ['SalePrice']].corr()
sns_hmap = seaborn.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")

# таблицы для записи итоговых результатов
result = pandas.DataFrame()
time_result = pandas.DataFrame()

# # ОБУЧЕНИЕ МОДЕЛЕЙ
# # Оригинальные данные
# af.get_analiz(data_num, df_target, result, time_result, 'original', 0)
af.scatter_plot_func(DataFrame, data_num, 'price_group', "ORIGINAL")
# # Стандартизация
df_stand = preprocessing.scale(data_num)
df_stand = pandas.DataFrame(data=df_stand, index=data_num.index, columns=attributes)
# af.get_analiz(df_stand, df_target, result, time_result, 'standardization', 1)
af.scatter_plot_func(DataFrame, df_stand, 'price_group', "STANDARDIZATION")
# # Нормализация
df_norm = preprocessing.normalize(data_num)
df_norm = pandas.DataFrame(data=df_norm, index=data_num.index, columns=attributes)

reduced_data = PCA(n_components=2).fit_transform(df_stand)
kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
kmeans.fit(reduced_data)
h = .05

# Граничные значения и значения сетки
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

print('x_min = ', x_min, 'x_max = ', x_max, 'y_min = ', y_min, 'y_max = ', y_max)
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
plt.title('K-means кластеризация базы рукописных цифр (PCA-reduced data)\n'
          'Центроиды отмечены белыми крестиками')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())


# af.get_analiz(df_norm, df_target, result, time_result, 'normalization', 2)
af.scatter_plot_func(DataFrame, df_norm, 'price_group', "NORMALIZATION")
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
