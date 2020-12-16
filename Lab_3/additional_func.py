import matplotlib.pyplot as plt
from matplotlib import cm
"""
Набор дополнительных функций для лабораторной работы 3 'Регрессия'
"""


"""
Функция для перевода категориального признкака в численный в лоб 
благодаря массиву с индексами
"""
def get_index(line, index_mas, unique_signs):
    for i in range(len(unique_signs)):
        if line == unique_signs[i]:
            return index_mas[i]

"""
Построение гистограммы распределения по входному признаку датафрейма
"""
def build_gistogramm(feature, data):
    plt.figure(num='plt.bar result' + feature)
    counts = data[feature].value_counts()
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=30)


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
