import matplotlib.pyplot as plt
import numpy, pandas
from timeit import default_timer as timer
from matplotlib import cm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression, BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import style
style.use('ggplot')

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
def apply_regression_method(model, X_train, y_train, X_test, y_test, result_table, time_result_table, it, type_data):
    degree = 4
    polynom_trans = PolynomialFeatures(degree=degree, include_bias=True)
    x2_train = polynom_trans.fit_transform(X=X_train)
    x2_test = polynom_trans.transform(X=X_test)

    # Обучение модели
    studying_time_start = timer()
    model.fit(x2_train, y_train)
    studying_time_stop = timer()

    # Предсказание
    predict_time_start = timer()
    pred_train = model.predict(X=x2_train)
    pred_test = model.predict(X=x2_test)
    predict_time_stop = timer()

    # Параметры и метрики модели
    score_train = r2_score(y_train, pred_train)
    score_test = r2_score(y_test, pred_test)
    mse_train = mean_squared_error(y_true=y_train, y_pred=pred_train) / X_train.shape[0]
    mse_test = mean_squared_error(y_true=y_test, y_pred=pred_test) / X_train.shape[0]
    mae_train = mean_absolute_error(y_true=y_train, y_pred=pred_train) / X_train.shape[0]
    mae_test = mean_absolute_error(y_true=y_test, y_pred=pred_test) / X_train.shape[0]
    msle_train = mean_squared_log_error(y_true=y_train, y_pred=pred_train)
    msle_test = mean_squared_log_error(y_true=y_test, y_pred=pred_test)
    print('r2 train: {:.3f}'.format(score_train))
    print('r2 test: {:.3f}'.format(score_test))
    print('mse train: {:.3f}'.format(mse_train))
    print('mse test: {:.3f}'.format(mse_test))
    print('msle train: {:.3f}'.format(msle_train))
    print('msle test: {:.3f}'.format(msle_test))


    df = pandas.DataFrame({'Actual': y_test, 'Predicted': pred_test})
    print(str(model), '  ', type_data, '  IT: ', it)
    print(df)

    plt.ioff()
    plt.show()
    # Оценка верного угадывания модели
    true_pred = 0
    for i in range(len(df)):
        if df.at[i, 'Actual'] == df.at[i, 'Predicted']:
            true_pred = true_pred + 1
    # формирование записи в таблице для анализа
    # label = str(model)[:str(model).find('(')]
    # result_table.loc[it, label] = true_pred / len(df)
    # result_table.loc[it, 'type'] = type_data
    # time_result_table.loc[it, (label + ' ' + 'studying_time')] = studying_time_stop - studying_time_start
    # time_result_table.loc[it, (label + ' ' + 'predict_time')] = predict_time_stop - predict_time_start
    # time_result_table.loc[it, 'type'] = type_data
    # print('alg: ', label, '  type = ', type_data, '  % = ', true_pred / len(df))


class PolynomialRegression():
    degr = 0
    model = 0
    polynom_trans = 0

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=None, degree=1):
        self.degr = degree
        self.model = LinearRegression()
        self.polynom_trans = PolynomialFeatures(degree=self.degr, include_bias=True)

    def fit(self, X_train, y_train):
        x_train = self.polynom_trans.fit_transform(X=X_train)
        self.model.fit(x_train, y_train)

    def predict(self, y_train):
        return self.model.predict(y_train)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, *params):
        self.degr = params.values()
        self.polynom_trans = PolynomialFeatures(degree=self.degr, include_bias=True)

"""
Фукнция, разбивающая датафрем на несколько наборов тренеровочных и тестовых фреймов,
применяющая эти фреймы к раличным методам классификации
"""
def get_analiz(data, df_target, result_table, time_result_table, type_data, num_meth):
    kf = KFold(n_splits=2, shuffle=True, random_state=12)
    for ikf, (train_index, test_index) in enumerate(kf.split(data)):
        X_train, X_test = data.values[train_index], data.values[test_index]
        y_train, y_test = df_target.values[train_index], df_target.values[test_index]

        # print('IT = ', ikf)
        # print('num_neth = ', num_meth)
        # LinearRegression(),
        # apply_regression_method(ElasticNet(alpha=0.2, fit_intercept=False),
        #                         X_train, y_train, X_test, y_test, result_table, time_result_table,
        #                         ikf + num_meth, type_data)

        # result_table.loc[5 * ikf+num_meth, 'it'] = ikf
        # time_result_table.loc[5 * ikf + num_meth, 'it'] = ikf

        # Жирный кусок для автоматического подбора параметров
        # для всех вышеприменненных методов регрессии
        grid_param = {'alpha': list(numpy.arange(0.05, 1, 0.05)),
                      'fit_intercept': ['False', 'True']}
        grid_search = RandomizedSearchCV(ElasticNet(), param_distributions=grid_param,
                                         n_iter=20, scoring='neg_median_absolute_error')
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(ElasticNet()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        grid_param = {'n_iter': list(range(1, 200, 5)),
                      'fit_intercept': ['False'],
                      'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      'alpha_1': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      'alpha_2': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      'lambda_1': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      'lambda_2': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
        grid_search = RandomizedSearchCV(BayesianRidge(), param_distributions=grid_param,
                                         n_iter=50, scoring='neg_median_absolute_error')
        grid_search.fit(X_train, y_train)
        print('type = ', type_data, '  methot = ', str(BayesianRidge()))
        print(grid_search.best_params_)
        print(grid_search.best_score_)