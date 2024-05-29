import numpy as np
import time
from newton_method_const import newton_with_constant_learning_rate
from mainLab2 import rosenbrock, grad_rosenbrock, hessian_rosenbrock
import optuna

def simulated_annealing(func, x0, temp=2.0, cool_rate=0.69, iter_per_temp=600):
    x = np.array(x0)
    history = [x]
    total_iterations = 0
    while temp > 1e-8:  # Добавлено условие для остановки
        for _ in range(iter_per_temp):
            total_iterations += 1
            new_x = x + np.random.normal(size=x.shape)
            delta = func(new_x) - func(x)
            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                x = new_x
                history.append(x)
        temp *= cool_rate
    return x, total_iterations, history


def calc_time(func, args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    taken_time = end_time - start_time
    return result + (taken_time,)  # Возвращаем результат вместе со временем


# Исходные данные
x0 = [100, 100]


# Метод Ньютона
min_x_newton, iters_newton, taken_time_newton, _ = newton_with_constant_learning_rate(rosenbrock, grad_rosenbrock, hessian_rosenbrock, x0)
print("Метод Ньютона:")
print("Минимальная точка:", min_x_newton)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(min_x_newton))
print("Время выполнения:", taken_time_newton)
print()

# Метод имитации отжига


# Параметры для имитации отжига
def generate_params():
    params = []
    for temp in range(1, 11):  # temp пробегает значения 1.0, 2.0 ... 10.0
        for cool_rate in range(19, 100, 10):  # cool_rate: 0.19, 0.29 ... 0.99
            for iter_per_temp in range(100, 600, 100):  # iter_per_temp: 100, 200 ... 1000
                params.append((temp, cool_rate / 100, iter_per_temp))
    return params


params = generate_params()
total_params = len(params)  # общее количество параметров

# Минимальное и минимальное время и параметры при которых оно было достигнуто
min_time = float('inf')
best_params = None
best_point = None
max_time = 0
worst_params = None
worst_point = None

for i, (temp, cool_rate, iter_per_temp) in enumerate(params):
    point, _, _, taken_time_sa = calc_time(simulated_annealing, (rosenbrock, x0, temp, cool_rate, iter_per_temp))
    if np.linalg.norm(point - min_x_newton) <= 1:  # проверка разницы между точками
        if taken_time_sa < min_time:
            min_time = taken_time_sa
            best_params = (temp, cool_rate, iter_per_temp)
            best_point = point

        if taken_time_sa > max_time:
            max_time = taken_time_sa
            worst_params = (temp, cool_rate, iter_per_temp)
            worst_point = point

    print(f"Посчитано для {i+1} параметров из {total_params}")

print("Метод имитации отжига:")
print(f"Параметры (temp={best_params[0]}, cool_rate={best_params[1]}, iter_per_temp={best_params[2]}):")
print("Минимальное время выполнения:", min_time)
print("Полученная точка:", best_point)
print("Значение функции Розенброка в минимальной точке:", rosenbrock(best_point))
print()
print(f"Параметры (temp={worst_params[0]}, cool_rate={worst_params[1]}, iter_per_temp={worst_params[2]}):")
print("Максимальное время выполнения:", max_time)
print("Полученная точка:", worst_point)
print("Значение функции Розенброка в максимальной точке:", rosenbrock(worst_point))

