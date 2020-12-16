import pandas, numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder


pandas.set_option('display.max_columns', None)
DataFrame = pandas.read_csv('AB_NYC_2019.csv')
DataFrame = DataFrame.dropna()

unique_type_of_room = DataFrame['room_type'].unique()
index_for_unique_type_of_room = [2, 3, 1]


def get_index(line, index_mas):
    for i in range(len(unique_type_of_room)):
        if line == unique_type_of_room[i]:
            return index_mas[i]


DataFrame.loc[:, 'type_of_room'] = \
    DataFrame['room_type'].apply(lambda x: get_index(str(x), index_for_unique_type_of_room))


#DataFrame.describe()
#DataFrame.mean()
#DataFrame.groupby()

attributes = ["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",\
              "number_of_reviews", "last_review", "reviews_per_month"]

scatter_matrix(DataFrame[attributes])

DataFrame = DataFrame.query("price > 0 and price < 800")
DataFrame = DataFrame.query("reviews_per_month < 30")
DataFrame = DataFrame.query("minimum_nights < 29")
DataFrame = DataFrame.query("calculated_host_listings_count < 50")


def build_gistogramm(feature):
    plt.figure(num='plt.bar result' + feature)
    counts_neighb_gr = DataFrame[feature].value_counts()
    plt.bar(counts_neighb_gr.index, counts_neighb_gr.values)
    plt.xticks(rotation=30)


build_gistogramm('neighbourhood_group')
build_gistogramm('room_type')
build_gistogramm('price')
build_gistogramm('minimum_nights')
build_gistogramm('reviews_per_month')

scatter_matrix(DataFrame[attributes])



data_cat = DataFrame["neighbourhood_group"]
data_cat_encoder, data_categories = data_cat.factorize()

encoder = OneHotEncoder()
data_cat_1hot = encoder.fit_transform(data_cat_encoder.reshape(-1, 1))

print(data_cat_1hot.toarray())

DataFrame.loc[:, ['name1', 'name2', 'name3', 'name4', 'name5']] = data_cat_1hot.toarray()

attributes = ["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",\
              "number_of_reviews", "last_review", "reviews_per_month", "type_of_room", \
              "name1", "name2", "name3", "name4", "name5"]

scatter_matrix(DataFrame[attributes])




plt.show()
