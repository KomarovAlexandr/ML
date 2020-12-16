import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
import additional_func as af
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', None)

dataset = pd.read_csv('AB_NYC_2019.csv')
print(dataset.info())
# Первые 5 строк датафрейма
# list_head = dataset.columns
# print(list_head)


"""
Подготовка набора данных (удаление строк с пустыпи элементами, удаление выбросов
"""
# Удаление нулей
data = dataset.dropna()

# Отображение всех уникальных значений признаков
# for i in list_head:
#     if dataset[i].dtype == object:
#         print(i, ": ")
#         print(dataset[i].unique())
#         print(dataset[i].value_counts())

print(data.describe())

a_series = (data != 0).any(axis=1)
new_df = data.loc[a_series]
# Удаление выбросов из цен
data = data[(data[["price"]] != 0).all(axis=1)]
data = data[(data[["price"]] < 800).all(axis=1)]
# Удаление строк с кол-вом отзывов в месяц больше 30
data = data[(data[["reviews_per_month"]] < 30).all(axis=1)]
# Удаление строк с минимальным кол-вом ночей больше 29
data = data[(data[["minimum_nights"]] < 29).all(axis=1)]
# Удаление стрко с кол-вом сдаваемого жилья на влажельца больше 50
data = data[(data[["calculated_host_listings_count"]] < 50).all(axis=1)]

print(data.describe())

# Перевод типа сдаваемых помещений из категориального призкнака в численный и
# добавление нового столбца
unique_type_of_room = data['room_type'].unique()
index_for_unique_type_of_room = [2, 3, 1]
data.loc[:, 'type_of_room'] = \
    data['room_type'].apply(lambda x: af.get_index(str(x),
                                                   index_for_unique_type_of_room, unique_type_of_room))

# Перевод района из категориального призкнака в численный и добавление нового столбца
# unique_type_of_neighbourhood_group = data['neighbourhood_group'].unique()
# index_for_unique_type_of_unique_type_of_neighbourhood_group = [3, 1, 4, 2, 5]
# data.loc[:, 'neighbourhood_group_num'] = \
#     data['neighbourhood_group'].apply(lambda x: af.get_index(str(x),
#                                                              index_for_unique_type_of_unique_type_of_neighbourhood_group,
#                                                              unique_type_of_neighbourhood_group))

# Перевод района из категориального признака в численный путем перевода в несколько новых унитарных признаков
# добавление новых столбцов
data_cat = data["neighbourhood_group"]
data_cat_encoder, data_categories = data_cat.factorize()
encoder = OneHotEncoder()
data_cat_1hot = encoder.fit_transform(data_cat_encoder.reshape(-1, 1))
data.loc[:, ['name1', 'name2', 'name3', 'name4', 'name5']] = data_cat_1hot.toarray()

# Преобразование цен из численого признка в категориальный и добавление нового столбца
labels = ['free', 'cheap', 'a little more expensive', 'average', 'expensive']
price_group = pd.cut(data['price'],
                    bins=[0, 5, 15, 45, 100, data['price'].max()],
                    labels=labels)
data_price = price_group
data.loc[:, 'data_price'] = data_price

# Преобразование категориий цен в численный признак
# labels_num = [0, 50, 100, 500, 700]
# price_group_num = pd.cut(data['price'],
#                     bins=[0, 50, 100, 300, 500, data['price'].max()],
#                     labels=labels_num)
# data_price_group_num = price_group_num

# ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
attributes = ["type_of_room", "number_of_reviews", "minimum_nights"]
# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = data[attributes]

# Вывод графиков корреляциий
signs_for_build_correlations = ['price'] + attributes
scatter_matrix(data[signs_for_build_correlations])

# Красивый вывод карты с объявлениями и цветом обозначена цена
sns_plot = sns.relplot(x="longitude", y="latitude",
                       hue='data_price', data=data)
sns_plot.fig.suptitle("scatter plot")


"""
Построение диаграммы рассеяния 
"""
df_stand = preprocessing.scale(data_num)
df_stand = pd.DataFrame(data=df_stand, index=data_num.index, columns=attributes)

# Метод главных компонент
pca = PCA(n_components=2)
pca.fit(df_stand)
data_reduced = pca.transform(df_stand)
# Нормализация
# min_max_skaler = preprocessing.MinMaxScaler()
# data_reduced = min_max_skaler.fit_transform(data_reduced)

# Построение диаграммы рассеяния
fig = plt.figure()
fig.suptitle("Scatter plot normalize")
labels = data_price.unique()
plots = []
colors = cm.rainbow(np.linspace(0, 1, len(labels)))

for ing, ng in enumerate(labels):
    plots.append(plt.scatter(x=data_reduced[data_price == ng, 0],
                            y=data_reduced[data_price == ng, 1],
                            c=colors[ing],
                            edgecolor='k'))
plt.xlabel("component1")
plt.ylabel("component2")
plt.legend(plots, labels, loc="lower right", title="species")

"""
Обчение модули каким-то из методов регрессии
"""
X_train, X_test, y_train, y_test = train_test_split(df_stand, data['price'], test_size=0.2, random_state=42)

# lin_reg = Ridge(alpha=1, solver="cholesky")
# lin_reg.fit(X_train, y_train)
# y_pred = lin_reg.predict(X_test)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=1)
elastic_net.fit(X_train, y_train)
y_pred = elastic_net.predict(X_test)

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df = pd.DataFrame()
df.loc[:, 'Actual'] = y_test.array
df.loc[:, 'Predicted'] = y_pred

labels = [0, 50, 100, 500, 700]
y_test_group = pd.cut(df['Actual'],
                    bins=[0, 5, 15, 45, 100, df['Actual'].max()],
                    labels=labels)
df.loc[:, 'Actual_group'] = y_test_group
y_pred_group = pd.cut(df['Predicted'],
                    bins=[df['Predicted'].min(), 5, 15, 45, 100, df['Predicted'].max()],
                    labels=labels)
df.loc[:, 'Predicted_group'] = y_pred_group

mas = []
true_mas = 0
false_mas = 0
for i in range(len(df)):
    if df.at[i, 'Actual_group'] == df.at[i, 'Predicted_group']:
        mas.append(True)
        true_mas = true_mas + 1
    else:
        mas.append(False)
        false_mas = false_mas + 1
df.loc[:, 'result'] = mas
print(true_mas / len(df))


# plt.figure(num="name")
# plt.plot(X_train, y_train, "b.")
# plt.plot(X_test, y_pred, "r.")


# print(elastic_net.intercept_, elastic_net.coef_)
# print(elastic_net.predict(X_test))
# print(y_test[:50])

plt.show()
