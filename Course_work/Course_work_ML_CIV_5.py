import pandas, numpy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import additional_func as af
from sklearn import preprocessing

data_frame = pandas.read_csv('Civ5 Leader Bias.csv')
pandas.set_option('display.max_columns', None)

# Запись все параметров во внешний файл
# Output_file = open('Output.txt', 'w', encoding="utf8")
# for line in DataFrame['Unnamed: 0'].to_list():
#     Output_file.write(line + '\n')
# pandas.reset_option('max_columns')

# Поворачиваем таблицу и устанавливаем индекс
data_frame = data_frame.set_index('Unnamed: 0')
data_frame = data_frame.transpose()

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

data_frame[Predicted_parameters].hist()
scatter_matrix(data_frame[resource_signs])
param = ['Culture', 'Diplomacy', 'Spaceship', 'Gold', 'Growth','Infrastructure', 'Production', 'Religion', 'Science', 'Boldness', 'War']
scatter_matrix(data_frame[param])

corr = data_frame[['Spaceship'] + data_frame.columns.to_list()].corr().iloc[0]
corr.sort_values().to_excel('correlation.xlsx')
# def build_gistogramm(feature):
#     plt.figure(num='plt.bar result ' + feature)
#     Values = DataFrame[feature].value_counts()
#     plt.bar(Values.index, Values.values)
#     plt.xticks(rotation=30)


# Разбиение цен на группы и создание нового столбца
labels = [1, 2, 3]
price_group = pandas.cut(data_frame['Spaceship'],
                    bins=[data_frame['Spaceship'].min(),
                          5,
                          7,
                          data_frame['Spaceship'].max()],
                    labels=labels)
data_frame['Spaceship'] = numpy.array(price_group)
af.build_bar_chart(data_frame, 'Spaceship')

labels = [1, 2, 3]
price_group = pandas.cut(data_frame['Diplomacy'],
                    bins=[data_frame['Diplomacy'].min(),
                          5,
                          6,
                          data_frame['Diplomacy'].max()],
                    labels=labels)
data_frame['Diplomacy'] = numpy.array(price_group)
af.build_bar_chart(data_frame, 'Diplomacy')

labels = [1, 2, 3]
price_group = pandas.cut(data_frame['Culture'],
                    bins=[data_frame['Culture'].min(),
                          5,
                          6,
                          data_frame['Culture'].max()],
                    labels=labels)
data_frame['Culture'] = numpy.array(price_group)
af.build_bar_chart(data_frame, 'Culture')

data_frame = data_frame.dropna()

# ИТОГОВЫЙ НАБОР ПРИЗНАКОВ
# for Culture
# attributes = ['Expansion', 'Military Training', 'Victory Competitiveness', 'War', 'Gold', 'CS Bully',
#               'Growth', 'Warmonger Hatred', 'Guarded', 'Religion', 'Friendly', 'CS Protect']
# for Diplomacy
# attributes = ['Air', 'Naval Tile Improvement', 'Chattiness', 'Meanness', 'Gold',
#               'Build Nuke', 'Deceptive', 'Neediness', 'Guarded', 'Wonder', 'Diplomatic Balance', 'Science']
# for Spaceship
attributes = ['Guarded', 'Wonder', 'Diplomatic Balance', 'Science',
              'Air', 'Naval Tile Improvement', 'Chattiness', 'Meanness', 'Gold']

# ДАТАФРЕЙМ СО ВСЕМИ ЧИСЛЕННЫМИ ПРИЗНАКАМИ
data_num = data_frame[attributes]
# ЦЕЛЕВОЙ ПРИЗНАК
df_target = data_frame['Spaceship']

# таблицы для записи итоговых результатов
result = pandas.DataFrame()
time_result = pandas.DataFrame()

# Оригинальные данные
af.get_analiz(data_num, df_target, result, time_result, 'original', 0)
af.scatter_plot_func(data_frame, data_num, 'Culture', "ORIGINAL")
# Стандартизация
df_stand = preprocessing.scale(data_num)
df_stand = pandas.DataFrame(data=df_stand, index=data_num.index, columns=attributes)
af.get_analiz(df_stand, df_target, result, time_result, 'standardization', 1)
af.scatter_plot_func(data_frame, df_stand, 'Culture', "STANDARDIZATION")
# Нормализация
df_norm = preprocessing.normalize(data_num)
df_norm = pandas.DataFrame(data=df_norm, index=data_num.index, columns=attributes)
af.get_analiz(df_norm, df_target, result, time_result, 'normalization', 2)
af.scatter_plot_func(data_frame, df_norm, 'Culture', "NORMALIZATION")

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
