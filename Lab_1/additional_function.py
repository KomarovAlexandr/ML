import matplotlib.pyplot as plt
import numpy
from matplotlib import cm


def get_number_in_future(str_from_df, future):
    for nb_of_param, param in enumerate(future):
        if param == str_from_df:
            return nb_of_param


def scatter_plot_func(df, data, target, name):
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


def build_bar_chart(df, target, name='', change_x_ticks=False, angle=30):
    plt.figure(num=name + target)
    plt.suptitle(name + target)
    counts = df[target].value_counts()
    plt.bar(counts.index, counts.values)
    x_ticks_mas = []
    if change_x_ticks:
        for i in numpy.arange(0, 1, 0.1):
            x_ticks_mas.append(counts.index[int(len(counts.index) * i)])
    plt.xticks(x_ticks_mas, rotation=angle)


def search_for_popular_words_in_message(data, most_occur_names):
    data = data.lower()
    number = 0
    for i in range(25):
        for word in data:
            if word == most_occur_names[i][0]:
                number = number + 1
    return number
