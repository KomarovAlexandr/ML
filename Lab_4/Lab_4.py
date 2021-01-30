import pandas, seaborn, numpy
import matplotlib.pyplot as plt
import additional_func as af
from sklearn import preprocessing
from sklearn.decomposition import PCA

import warnings
warnings.simplefilter('ignore')

pandas.set_option('display.max_columns', None)
DataFrame = pandas.read_csv('train.csv')

# Получение таблицы с признаками, их типом, максимальным и минимальным значениями
DataFrame.describe().to_excel('data_describe.xlsx')

# DataFrame = DataFrame[(DataFrame[["Id"]] % 2 == 0).all(axis=1)]

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
DataFrame = DataFrame[(DataFrame[["SalePrice"]] < 300000).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["YearBuilt"]] > 1910).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["GrLivArea"]] > 800).all(axis=1)]
DataFrame = DataFrame[(DataFrame[["GrLivArea"]] < 2200).all(axis=1)]

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
labels = [1, 0, 2, 0, 3]
delta_price = (DataFrame['SalePrice'].max() - DataFrame['SalePrice'].min()) / 5
print(DataFrame['SalePrice'].min(), ' ', delta_price, ' ', DataFrame['SalePrice'].max())
price_group = pandas.cut(DataFrame['SalePrice'],
                    bins=[DataFrame['SalePrice'].min(),
                          DataFrame['SalePrice'].min() + 2.0*delta_price,
                          DataFrame['SalePrice'].min() + 2.2*delta_price,
                          DataFrame['SalePrice'].min() + 2.8*delta_price,
                          DataFrame['SalePrice'].min() + 3.0*delta_price,
                          DataFrame['SalePrice'].max()],
                    labels=labels, ordered=False)
DataFrame.loc[:, 'price_group'] = numpy.array(price_group)
DataFrame = DataFrame[(DataFrame[['price_group']] > 0).all(axis=1)]
DataFrame.reset_index(drop=True)

# Дополнительное удаление пустых строк. Почему то это нужно сделать еще раз, иначе бо-бо
DataFrame = DataFrame.dropna()

# ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
attributes = ['PricePerFoot', 'OverallQual', 'GrLivArea', 'YearBuilt', 'SalePrice']
# attributes = ['price_group', 'OverallQual']
# attributes = ['PricePerFoot', 'OverallQual']
# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = DataFrame[attributes]

# Вывод всех гистограмм
# for i in attributes + ['price_group']:
#     af.build_bar_chart(DataFrame, i, i, )

# Построение карты корреляций
plt.figure()
corr = DataFrame.corr()
sns_hmap = seaborn.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")
# corr = DataFrame.corr().iloc[0]
corr.to_excel('corr.xlsx')

# таблицы для записи итоговых результатов
result = pandas.DataFrame()
time_result = pandas.DataFrame()

# ОБУЧЕНИЕ МОДЕЛЕЙ
# Оригинальные данные
af.scatter_plot_func(DataFrame, data_num, 'price_group', "ORIGINAL")
# Нормализация
df_norm = PCA(n_components=2).fit_transform(data_num)
df_norm = preprocessing.normalize(data_num)
df_norm = pandas.DataFrame(data=df_norm, index=data_num.index, columns=attributes)
af.scatter_plot_func(DataFrame, df_norm, 'price_group', "NORMALIZATION")


reduced_data = PCA(n_components=2).fit_transform(data_num)
plt.figure()
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
labels_true = DataFrame['price_group']

# reduced_data = df_stand.to_numpy()
'''--------------------------K-means-----------------------------------------'''
from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=50)
af.show_klasters(reduced_data, kmeans, DataFrame['price_group'], 'K-means')


'''-----------------------GaussianMixture-----------------------------------'''
from sklearn.mixture import GaussianMixture
model_GaussianMixture = GaussianMixture(n_components=3, covariance_type='spherical', tol=1e-3, reg_covar=1e-6, max_iter=1000,
                                        init_params='kmeans', random_state=0, n_init=5)
af.show_klasters(reduced_data, model_GaussianMixture, DataFrame['price_group'], 'GaussianMixture')


'''-------------------------DBSCAN--------------------------------------'''
from sklearn.cluster import DBSCAN
model_dbscan = DBSCAN(eps=8000, min_samples=2)
af.show_klasters(reduced_data, model_dbscan, DataFrame['price_group'], 'DBSCAN')


'''-------------------------SpectralClustering--------------------------------------'''
from sklearn.cluster import SpectralClustering
reduced_data = PCA(n_components=2).fit_transform(df_norm)
model_spectClust = SpectralClustering(n_clusters=3, assign_labels="kmeans", gamma=10,
                                      random_state=25, n_neighbors=8, affinity='nearest_neighbors')
af.show_klasters(reduced_data, model_spectClust, DataFrame['price_group'], 'SpectralClustering')

plt.show()
pandas.reset_option('max_columns')

