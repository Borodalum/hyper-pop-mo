import numpy as np

from newton_cg_quasi import newton_cg, quasinewton
from newton_method_const import newton_with_constant_learning_rate
from newton_method_search import newton_with_search_learning_rate
from newton_wolfe import newton_with_wolfe_search
from plotting import plot_graphs
import pandas as pd


# Функция Розенброка
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# Градиент функции Розенброка
def grad_rosenbrock(x):
    return np.array([400 * x[0] ** 3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1] - x[0] ** 2)])


# Гессиан функции Розенброка
def hessian_rosenbrock(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200]])


# Неполиномиальная функция
def non_polynomial_function(x):
    return x[0] ** 2 + x[1] ** 2 + np.sin(x[0]) * np.cos(x[1])


# Градиент неполиномиальной функции
def grad_non_polynomial_function(x):
    df_dx = 2 * x[0] + np.cos(x[0]) * np.cos(x[1])
    df_dy = 2 * x[1] - np.sin(x[0]) * np.sin(x[1])
    return np.array([df_dx, df_dy])


# Гессиан неполиномиальной функции
def hessian_non_polynomial_function(x):
    d2f_dx2 = -np.sin(x[0]) * np.cos(x[1]) - np.sin(x[0]) * np.sin(x[1])
    d2f_dy2 = -np.cos(x[0]) * np.cos(x[1]) - np.cos(x[0]) * np.sin(x[1])
    d2f_dx_dy = -np.cos(x[0]) * np.sin(x[1]) + np.sin(x[0]) * np.cos(x[1])
    return np.array([[d2f_dx2, d2f_dx_dy], [d2f_dx_dy, d2f_dy2]])