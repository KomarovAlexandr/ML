import matplotlib.pyplot as plt
import numpy, pandas
from timeit import default_timer as timer
from matplotlib import cm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

"""
Построение гистограммы распределения по входному признаку датафрейма
"""
def build_bar_chart(df, target, name='', change_x_ticks=False, angle=30):
    plt.figure(num=name + target)
    plt.suptitle(name + target)
    counts = df[target].value_counts()
    plt.bar(counts.index, counts.values)
    x_ticks_mas = []
    if change_x_ticks:
        for i in numpy.arange(0, 1, 0.1):
            x_ticks_mas.append(counts.index[int(len(counts.index) * i)])
            plt.xticks(x_ticks_mas)
    plt.xticks(rotation=angle)


"""
Функция построения диаграммы рассеяния
"""
def scatter_plot_func(df, data_num, target, name):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data_num)
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


"""
Функция применяющая один из методов классификации и возвращающая процент верного предсказания
модели на тестовой выборке
"""
def apply_clustering_method(model, X_train, y_train, X_test, y_test, result_table, time_result_table, it, type_data):
    # Обучение модели
    studying_time_start = timer()
    model.fit(X_train, y_train)
    studying_time_stop = timer()
    # Предсказание
    predict_time_start = timer()
    y_pred = model.predict(X_test)
    predict_time_stop = timer()
    # Оценка верного угадывания модели
    df = pandas.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    true_pred = 0
    for i in range(len(df)):
        if df.at[i, 'Actual'] == df.at[i, 'Predicted']:
            true_pred = true_pred + 1
    # формирование записи в таблице для анализа
    label = str(model)[:str(model).find('(')]
    result_table.loc[it, label] = true_pred / len(df)
    result_table.loc[it, 'type'] = type_data
    # result_table.loc[it, 'studying_time'] = studying_time_stop - studying_time_start
    # result_table.loc[it, 'predict_time'] = predict_time_stop - predict_time_start
    time_result_table.loc[it, (label + ' ' + 'studying_time')] = studying_time_stop - studying_time_start
    time_result_table.loc[it, (label + ' ' + 'predict_time')] = predict_time_stop - predict_time_start
    time_result_table.loc[it, 'type'] = type_data
    # print('alg: ', label, '  type = ', type_data, '  % = ', true_pred / len(df))


"""
Фукнция, разбивающая датафрем на несколько наборов тренеровочных и тестовых фреймов,
применяющая эти фреймы к раличным методам классификации
"""
def get_analiz(data, df_target, result_table, time_result_table, type_data, num_meth):
    kf = KFold(n_splits=3, shuffle=True, random_state=12)
    for ikf, (train_index, test_index) in enumerate(kf.split(data)):
        X_train, X_test = data.values[train_index], data.values[test_index]
        y_train, y_test = df_target.values[train_index], df_target.values[test_index]
        # print('IT = ', ikf)
        # print('num_neth = ', num_meth)
        # apply_clustering_method(KNeighborsClassifier(n_neighbors=35, algorithm='brute', p=2),
        #                         X_train, y_train, X_test, y_test, result_table,
        #                         time_result_table, 5*ikf + num_meth, type_data)
        #
        # apply_clustering_method(GaussianNB(var_smoothing=0.075),
        #                         X_train, y_train, X_test, y_test, result_table,
        #                         time_result_table, 5*ikf + num_meth, type_data)
        #
        # apply_clustering_method(DecisionTreeClassifier(criterion='gini', min_samples_split=10,
        #                                                max_depth=20),
        #                         X_train, y_train, X_test, y_test, result_table,
        #                         time_result_table, 5*ikf + num_meth, type_data)
        #
        # apply_clustering_method(Pipeline([("scaller", StandardScaler()),
        #                                   ("svm_clf", SVC(kernel="rbf", gamma=3, C=10))]),
        #                         X_train, y_train, X_test, y_test, result_table,
        #                         time_result_table, 5*ikf + num_meth, type_data)
        #
        # apply_clustering_method(RandomForestClassifier(n_estimators=40, criterion='gini',
        #                                                min_samples_split=6, max_depth=40),
        #                         X_train, y_train, X_test, y_test, result_table,
        #                         time_result_table, 5*ikf + num_meth, type_data)
        # result_table.loc[5 * ikf+num_meth, 'it'] = ikf
        # time_result_table.loc[5 * ikf + num_meth, 'it'] = ikf

        # Жирный кусок для автоматического подбора параметров
        # для всех вышеприменненных методов классификации
        print('num_meth = ', num_meth, '  It = ', ikf)

        grid_param = {'n_neighbors': list(range(2, 20)),
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1, 2, 3]}
        grid_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=grid_param, n_iter=30)
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(KNeighborsClassifier()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        grid_param = {'var_smoothing': list(numpy.arange(0, 1, 0.01))}
        grid_search = RandomizedSearchCV(GaussianNB(), param_distributions=grid_param, n_iter=30)
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(GaussianNB()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        grid_param = {'criterion': ['gini', 'entropy'],
                      'min_samples_split': list(range(2, 12, 1)),
                      'max_depth': list(range(1, 100, 1))}
        grid_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=grid_param, n_iter=30)
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(DecisionTreeClassifier()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        grid_param = {'kernel': ["rbf"],
                      'gamma': numpy.arange(0.1, 5, 0.1),
                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = RandomizedSearchCV(SVC(), param_distributions=grid_param, n_iter=30)
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(SVC()), '  rbf')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        grid_param = {'n_estimators': range(2, 50, 2),
                      'criterion': ['gini', 'entropy'],
                      'min_samples_split': list(range(2, 12, 1)),
                      'max_depth': list(range(1, 100, 1))}
        grid_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=grid_param, n_iter=30)
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(RandomForestClassifier()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)

