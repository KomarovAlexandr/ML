import pandas, seaborn, numpy
import matplotlib.pyplot as plt
import additional_func as af
from sklearn.decomposition import PCA
from sklearn import preprocessing

pandas.set_option('display.max_columns', None)
data_frame = pandas.read_csv('train.csv')

# Получение таблицы с признаками, их типом, максимальным и минимальным значениями
data_frame.describe().to_excel('data_describe.xlsx')

# Удаление столбцов с большим кол-вом пустых значений
data_frame = data_frame.drop(['PoolQC', 'Alley', 'FireplaceQu', 'Fence', 'MiscFeature', 'LotFrontage'], axis=1)
# Фильтрация выбросов
data_frame = data_frame[(data_frame[["GrLivArea"]] < 4000).all(axis=1)]
data_frame = data_frame[(data_frame[["GarageCars"]] < 4).all(axis=1)]
data_frame = data_frame[(data_frame[["OverallQual"]] > 2).all(axis=1)]
data_frame = data_frame[(data_frame[["1stFlrSF"]] < 2000).all(axis=1)]
data_frame = data_frame[(data_frame[["GrLivArea"]] < 2500).all(axis=1)]
data_frame = data_frame[(data_frame[["TotalBsmtSF"]] < 2000).all(axis=1)]
data_frame = data_frame[(data_frame[["TotalBsmtSF"]] > 450).all(axis=1)]
data_frame = data_frame[(data_frame[["GarageArea"]] < 1000).all(axis=1)]


# Удаление нулей и обновление индексов
data_frame = data_frame.dropna()
data_frame.reset_index(drop=True)

print("Dataframe size = ", len(data_frame))

pca = PCA(n_components=1)
for feature in data_frame.columns:
    if data_frame[feature].dtype == object:
        data_frame[feature] = pandas.Categorical(data_frame[feature])
        sex_map_train = dict(zip(data_frame[feature].cat.codes, data_frame[feature]))
        data_frame[feature] = data_frame[feature].cat.codes
        # Ниже мой вариант того же самого что делают три строки выше
        # data_cat_encoder, data_categories = data_frame[feature].factorize()
        # encoder = OneHotEncoder()
        # data_cat_1hot = encoder.fit_transform(data_cat_encoder.reshape(-1, 1))
        # labels = []
        # for i in range(data_cat_1hot.toarray().shape[1]):
        #     labels.append('name ' + str(i))
        #
        # data = pandas.DataFrame(data=data_cat_1hot.toarray(), columns=labels)
        # data_frame[feature] = pca.fit_transform(data)

# Вывод значений корреляции всех признаков с ценой в табличку
corr = data_frame[['SalePrice'] + data_frame.columns.to_list()].corr().iloc[0]
corr.sort_values().to_excel('correlation.xlsx')

# Жирненький кусок для отображения всех-всех-всех корреляций
# На начальном анализа датафрейма пригодилось
# data_1 = ['SalePrice'] + data_frame.columns[1:15].to_list()
# data_2 = ['SalePrice'] + data_frame.columns[16:30].to_list()
# data_3 = ['SalePrice'] + data_frame.columns[31:45].to_list()
# data_4 = ['SalePrice'] + data_frame.columns[46:60].to_list()
# data_5 = ['SalePrice'] + data_frame.columns[61:70].to_list()
# data_6 = ['SalePrice'] + data_frame.columns[71:].to_list()
#
# corr_1 = data_frame[data_1].corr()
# corr_2 = data_frame[data_2].corr()
# corr_3 = data_frame[data_3].corr()
# corr_4 = data_frame[data_4].corr()
# corr_5 = data_frame[data_5].corr()
# corr_6 = data_frame[data_6].corr()
#
# plt.figure()
# seaborn.heatmap(corr_1).set_title("correlation 1 PANDAS + SEABORN")
# plt.figure()
# seaborn.heatmap(corr_2).set_title("correlation 2 PANDAS + SEABORN")
# plt.figure()
# seaborn.heatmap(corr_3).set_title("correlation 3 PANDAS + SEABORN")
# plt.figure()
# seaborn.heatmap(corr_4).set_title("correlation 4 PANDAS + SEABORN")
# plt.figure()
# seaborn.heatmap(corr_5).set_title("correlation 5 PANDAS + SEABORN")
# plt.figure()
# seaborn.heatmap(corr_6).set_title("correlation 6 PANDAS + SEABORN")

# Разбиение цен на группы и создание нового столбца
labels = [1, 2, 3]
price_group = pandas.cut(data_frame['SalePrice'],
                    bins=[data_frame['SalePrice'].min(),
                          145000,
                          200000,
                          data_frame['SalePrice'].max()],
                    labels=labels)
data_frame.loc[:, 'price_group'] = numpy.array(price_group)
# af.build_bar_chart(DataFrame, 'price_group')

# Дополнительное удаление пустых строк. Почему то это нужно сделать еще раз, иначе бо-бо
data_frame = data_frame.dropna()
# ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
attributes = ['GarageCars', 'GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageArea', '1stFlrSF',
              'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
              'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 'HeatingQC', 'GarageType']
# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = data_frame[attributes]
# ЦЕЛЕВОЙ ПРИЗНАК
df_target = data_frame['price_group']

# Отрисовка гистограмм всех признаков из итогового набора
# for i in attributes:
#     af.build_bar_chart(data_num, i)

# Построение карты корреляций
plt.figure()
corr = data_frame[attributes + ['price_group']].corr()
sns_hmap = seaborn.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")

# таблицы для записи итоговых результатов
result = pandas.DataFrame()
time_result = pandas.DataFrame()

# ОБУЧЕНИЕ МОДЕЛЕЙ
# Оригинальные данные
af.get_analiz(data_num, df_target, result, time_result, 'original', 0)
af.scatter_plot_func(data_frame, data_num, 'price_group', "ORIGINAL")
# Стандартизация
df_stand = preprocessing.scale(data_num)
df_stand = pandas.DataFrame(data=df_stand, index=data_num.index, columns=attributes)
af.get_analiz(df_stand, df_target, result, time_result, 'standardization', 1)
af.scatter_plot_func(data_frame, df_stand, 'price_group', "STANDARDIZATION")
# Нормализация
df_norm = preprocessing.normalize(data_num)
df_norm = pandas.DataFrame(data=df_norm, index=data_num.index, columns=attributes)
af.get_analiz(df_norm, df_target, result, time_result, 'normalization', 2)
af.scatter_plot_func(data_frame, df_norm, 'price_group', "NORMALIZATION")

# обновление индексации таблиц для корректного отображения
result = result.sort_values(['it']).set_index(['it', 'type'], drop=True)
time_result = time_result.sort_values(['it']).set_index(['it', 'type'], drop=True)

# Вывод в таблицы excel
result.to_excel('result.xlsx')
time_result.to_excel('time_result.xlsx')

print(result)
print(time_result)

plt.show()
pandas.reset_option('max_columns')