# -*- coding: utf-8 -*-
import pandas, numpy, folium
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from folium.plugins import HeatMap
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold
import additional_function as af

DataFrame = pandas.read_csv('AB_NYC_2019.csv')
pandas.set_option('display.max_columns', None)
PATH = 'C:/Users/al.kom/YandexDisk-raptor.al.com@yandex.ru/Machine_Learning/Lab_1/'

DataFrame = DataFrame.dropna()
DataFrame.reset_index()

"""
Part 1
"""
"""
1. Сколько записей в базе?
"""
num_rec_table = DataFrame.shape[0]  # Кол-во записей в таблице
print('num_rec_table = ', num_rec_table)

"""
2. Какие типы жилья и районы присутствуют в базе?
"""
type_of_room = DataFrame['room_type'].unique()  # Все типы жилья
type_of_neighbourhood_group = DataFrame['neighbourhood_group'].unique()  # Все типы районов
print('type_of_room = ', type_of_room)
print('type_of_neighbourhood_group = ', type_of_neighbourhood_group)

"""
3. Вычислите набор статистических значений цены для каждого типа жилья.
"""
room_type_stat = DataFrame.groupby(['room_type']).describe()['price']
print('room_type_stat', '\n', room_type_stat)


"""
4. Вычислите н набор статистических значений цены для каждого района.
"""
neighbourhood_group_type_stat = DataFrame.groupby(['neighbourhood_group']).describe()['price']
neighbourhood_group_type_stat.to_excel('4.xlsx')
print('neighbourhood_group_type_stat', '\n', neighbourhood_group_type_stat)

"""
5. Какова средняя стоимость жилья в зависимости от типа жилья и района?
"""
mean_price = pandas.pivot_table(DataFrame,
                         columns='room_type',
                         index='neighbourhood_group',
                         values='price',
                         aggfunc=numpy.mean)
print('mean_price', '\n', mean_price)
mean_price.to_excel('5.xlsx')

"""
6. Каково максимальное значение минимального количества ночей?
"""
max_minimum_nights = numpy.max(DataFrame['minimum_nights'])
print('max_minimum_nights = ', max_minimum_nights)

"""
7. Каковы минимальная, максимальная и средняя цена аренды жилья в
трёх самых дорогих соседствах Бруклина
"""
# Выкавыриваем все записи по соседству с Бруклином, сортируем по цене и забираем первые три записи
expensive_neighbourhood_group = DataFrame[DataFrame['neighbourhood_group'] == 'Brooklyn'].sort_values('price', ascending = False)['neighbourhood'][0:3]
# Для каждого района массив типа neighb_gr_X = [min_price, mean_price, max_price]
neighb_gr_1 = [0, 0, 0]
neighb_gr_2 = [0, 0, 0]
neighb_gr_3 = [0, 0, 0]


def prices_for_district(mas_of_prices, number_of_dist):
    # массив хранящий цены для следующей команды иначе больно длинно и жирно получается
    array_of_prices = DataFrame[DataFrame['neighbourhood'] == expensive_neighbourhood_group.array[number_of_dist]]['price'].array
    # выковариваем минимальное значение исключае нулевые
    mas_of_prices[0] = numpy.min(array_of_prices[numpy.nonzero(array_of_prices)])
    mas_of_prices[1] = numpy.mean(DataFrame[DataFrame['neighbourhood'] == expensive_neighbourhood_group.array[number_of_dist]]['price'])
    mas_of_prices[2] = numpy.max(DataFrame[DataFrame['neighbourhood'] == expensive_neighbourhood_group.array[number_of_dist]]['price'])


prices_for_district(neighb_gr_1, 0)
prices_for_district(neighb_gr_2, 1)
prices_for_district(neighb_gr_3, 2)
print('neighb_gr_1 = ', neighb_gr_1)
print('neighb_gr_2 = ', neighb_gr_2)
print('neighb_gr_3 = ', neighb_gr_3)

"""
8. Составьте рейтинг слов из названий по популярности (частоте появления)
и укажите 25 самых популярных с числом их появлений
"""
# Преобразуем стобец фрейма в массив, опускаем в нижний регистр и сортируем
names_list = DataFrame['name'].to_list()
names = ''
for name in names_list:
    names = names + str(name) + ' '
counter_mas = Counter(names.lower().split()).most_common(35)
print(counter_mas)
most_occur_names = []
Output_file = open(PATH + 'Output.txt', 'w', encoding="utf8")
for word in counter_mas:
    if len(word[0]) > 1:
        Output_file.write(word[0] + '   ' + str(word[1]) + '\n')
        most_occur_names.append(str(word[0]))
    if len(most_occur_names) >= 25:
        break
Output_file.close()
print('len(most_occur_names) = ', len(most_occur_names))
print('most_occur_names = ', most_occur_names)
"""
# Part 2
"""
"""
9. Постройте график (диаграмму рассеяния, точечную диаграмму), где по осям х и у
будут широта и долгота, а цветом помечены районы
"""
sns_plot = sns.relplot(x="longitude", y="latitude",
                       hue='neighbourhood_group', data=DataFrame)
sns_plot.fig.suptitle("scatter plot")

m = folium.Map([40.7128, -74.0060], zoom_start=11)
HeatMap(DataFrame[['latitude', 'longitude']].dropna(), radius=8,
        gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1.0: 'red'}).add_to(m)
m.save('map.html')

"""
10. Постройте гистограммы всех признаков, для которых есть смысл это делать.
"""
af.build_bar_chart(DataFrame, 'neighbourhood_group', change_x_ticks=False, angle=0)

af.build_bar_chart(DataFrame, 'last_review', change_x_ticks=True, angle=25)

af.build_bar_chart(DataFrame, 'reviews_per_month', change_x_ticks=False, angle=30)

DataFrame = DataFrame[(DataFrame[["minimum_nights"]] < 29).all(axis=1)]
af.build_bar_chart(DataFrame, 'minimum_nights', change_x_ticks=False, angle=30)

"""
11. Из названий получите три новых признака (добавьте столбцы в таблицу): длина
названия в символах len_name , число слов nb_words , число популярных слов
nb_pop_words (из пункта 8).
"""
DataFrame.loc[:, 'len_name'] = DataFrame['name'].apply(lambda x: len(str(x)))
DataFrame.loc[:, 'number_of_word'] = DataFrame['name'].apply(lambda x: len(str(x).split()))
DataFrame.loc[:, 'number__of_popular_word'] = \
    DataFrame['name'].apply(lambda x: af.search_for_popular_words_in_message(str(x), most_occur_names))

"""
12. Преобразуйте цену из численного типа в категориальный (пока неважно, сколько 
будет категорий, возьмите что-то между 3 и 10). Добавьте преобразованный вариант 
цены в таблицу, пусть это будет признак “price_group ”.
"""
column_without_nan = DataFrame['price'].fillna(-1)
labels = ['0 - 10', '10 - 20', '20 - 40', '40 - 80',  '80 - 160', '160 - 320', '320 - 640', '640 - 800']
price_group = pandas.cut(column_without_nan,
                    bins=[0, 10, 20, 40, 80, 160, 320, 640, DataFrame['price'].max()],
                    labels=labels)
DataFrame.loc[:, 'price_group'] = price_group

plt.figure(num="plt.bar result (price_group)")
counts = DataFrame['price_group'].value_counts()
plt.bar(counts.index, counts.values)
plt.xticks(rotation=30)

"""
13. Примените метод главных компонент, чтобы преобразовать вектор признаков в 
двумерный. Приведите диаграмму рассеяния, в которой по осям будут получившиеся 
компоненты, а целевая переменная будет отмечена цветом.
"""
DataFrame.loc[:, 'type_of_neighbourhood_group'] = \
    DataFrame['neighbourhood_group'].apply(lambda x: af.get_number_in_future(x, type_of_neighbourhood_group))
DataFrame.loc[:, 'type_of_room'] = \
    DataFrame['room_type'].apply(lambda x: af.get_number_in_future(x, type_of_room))

pca = PCA(n_components = 2)
df_labels = ['type_of_room', 'latitude', 'longitude', 'price',
             'len_name', 'number_of_word', 'number__of_popular_word']
df = DataFrame[df_labels]
pca.fit(df)
data_reduced = pca.transform(df)
af.scatter_plot_func(DataFrame, data_reduced, 'neighbourhood_group', "Scatter plot")

"""
14. Нормализуйте все признаки. Снова примените метод главных компонент и
постройте диаграмму. Затем стандартизируйте исходные признаки, примените 
метод главных компонент и снова постройте диаграмму.
"""
df_norm = preprocessing.normalize(df)
pca.fit(df_norm)
data_reduced = pca.transform(df_norm)
af.scatter_plot_func(DataFrame, data_reduced, 'neighbourhood_group', "Scatter plot NORM")

df_stand = preprocessing.scale(df)
pca.fit(df_stand)
data_reduced = pca.transform(df_stand)
af.scatter_plot_func(DataFrame, data_reduced, 'neighbourhood_group', "Scatter plot SCALE")

"""
15. Постройте матрицу корреляции, но, пожалуйста, с цветовой индикацией.
"""
plt.figure()
corr = df.corr()
sns_hmap = sns.heatmap(corr)
sns_hmap.set_title("correlation PANDAS + SEABORN")

"""
# Part 3
"""
"""
16. Выполните 3 разбиения для базы из п.13 (любой вариант - исходный,
нормализованный, стандартизированный) c помощью генератора разбиений
KFold . Для каждого разбиения приведите гистограммы цен и районов.
( презентация, слайд 22 )
"""
df_stand = pandas.DataFrame({df_labels[0]: df_stand[:, 0], df_labels[1]: df_stand[:, 1],
                             df_labels[2]: df_stand[:, 2], df_labels[3]: df_stand[:, 3],
                             df_labels[4]: df_stand[:, 4], df_labels[5]: df_stand[:, 5],
                             df_labels[6]: df_stand[:, 6]})
df_target = pandas.DataFrame({'neighbourhood_group': DataFrame['neighbourhood_group']})

print('PART 3')
kf = KFold(n_splits=3, shuffle=False, random_state=None)
for ikf, (train_index, test_index) in enumerate(kf.split(df_stand)):
    X_train, X_test = df_stand.to_numpy()[train_index], df_stand.to_numpy()[test_index]
    y_train, y_test = df_target.to_numpy()[train_index], df_target.to_numpy()[test_index]
    print('Iteration: ', ikf)
    print(X_train, y_train)
    print(X_test, y_test)
    af.build_bar_chart(df_stand, 'price', name=str(ikf) + ' ', change_x_ticks=False, angle=0)
    af.build_bar_chart(df_target, 'neighbourhood_group', name=str(ikf) + ' ', change_x_ticks=False, angle=0)

plt.show()
pandas.reset_option('max_columns')
