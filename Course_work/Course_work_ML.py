import pandas, numpy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
DataFrame = pandas.read_csv('Civ5 Leader Bias.csv')
pandas.set_option('display.max_columns', None)

# Запись все параметров во внешний файл
# Output_file = open('Output.txt', 'w', encoding="utf8")
# for line in DataFrame['Unnamed: 0'].to_list():
#     Output_file.write(line + '\n')
# pandas.reset_option('max_columns')

# Поворачиваем таблицу и устанавливаем индекс
DataFrame = DataFrame.set_index('Unnamed: 0')
DataFrame = DataFrame.transpose()

# Выбор признаков касаемых добычи ресурсов
resource_signs = ['Archaeology', 'Expansion', 'Gold', 'Great People', 'Growth', 'Happiness', 'Infrastructure',
                  'Military Training', 'Production', 'Religion', 'Science', 'Wonder']
# Выбор признаков черт характера персонажа
character_signs = ['Boldness', 'Chattiness', 'Denounce Willingness', 'Diplomatic Balance', 'Friendship Willingness',
                   'Forgiveness', 'Loyalty', 'Meanness', 'City-State Competitiveness', 'Warmonger Hatred',
                   'Wonder Competitiveness', 'Afraid', 'Deceptive', 'Friendly', 'Guarded', 'Hostile', 'Neutrality',
                   'War']
# Предсказываемые параметры
Predicted_parameters = ['Culture', 'Diplomacy', 'Spaceship']

DataFrame[Predicted_parameters].hist()
scatter_matrix(DataFrame[resource_signs])
param = ['Culture', 'Diplomacy', 'Spaceship', 'Gold', 'Growth','Infrastructure', 'Production', 'Religion', 'Science', 'Boldness', 'War']
scatter_matrix(DataFrame[param])

def build_gistogramm(feature):
    plt.figure(num='plt.bar result ' + feature)
    Values = DataFrame[feature].value_counts()
    plt.bar(Values.index, Values.values)
    plt.xticks(rotation=30)

Predicted_parameter = DataFrame['Spaceship']

# Метод главных компонент
from sklearn.decomposition import PCA
data_num = DataFrame[resource_signs + character_signs]
pca = PCA(n_components=2)
pca.fit(data_num)
data_reduced = pca.transform(data_num)
# Нормализация (по идее ничего не меняет)
# from sklearn import preprocessing
# min_max_skaler = preprocessing.MinMaxScaler()
# data_reduced = min_max_skaler.fit_transform(data_reduced)

# Построение диаграммы рассеяния
from matplotlib import cm
import numpy
fig = plt.figure()
fig.suptitle("Scatter plot normalize")
plots = []
labels = numpy.sort(Predicted_parameter.unique())
colors = cm.rainbow(numpy.linspace(0, 1, len(labels)))

for ing, ng in enumerate(labels):
    plots.append(plt.scatter(x=data_reduced[Predicted_parameter == ng, 0],
                            y=data_reduced[Predicted_parameter == ng, 1],
                            c=colors[ing],
                            edgecolor='k'))
plt.xlabel("component1")
plt.ylabel("component2")
plt.legend(plots, labels, loc="lower right", title="species")

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data_reduced, Predicted_parameter, test_size=0.2, random_state=42)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# lin_reg = Ridge(alpha=1, solver="cholesky")
# lin_reg.fit(X_train, y_train)
#
# y_pred = lin_reg.predict(X_test)
#
# df = pandas.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print("result: ")
# print(df)
#
# plt.figure(num="name")
# plt.plot(X_train, y_train, "b.")
# plt.plot(X_test, y_pred, "r-")
#
# print(lin_reg.intercept_, lin_reg.coef_)
# print(lin_reg.predict(X_test))
# print(y_test[:50])
#
plt.show()
