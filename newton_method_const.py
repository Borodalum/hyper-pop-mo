import numpy as np

from calc_time_hyperparam import calc_time


def newton_with_constant_learning_rate(f, grad_f, hessian_f, x0, learning_rate=0.1, epsilon=1e-8, max_iter=1000):
    return calc_time(minimize, (f, grad_f, hessian_f, x0, learning_rate, epsilon, max_iter))


def minimize(f, grad_f, hessian_f, x0, learning_rate, epsilon, max_iter):
    x = np.array(x0)
    all_points = [x]
    iter_count = 0
    while iter_count < max_iter:
        grad = grad_f(x)
        if np.linalg.norm(grad) < epsilon:
            break
        delta_x = -np.linalg.inv(hessian_f(x)).dot(grad) * learning_rate
        x = x + delta_x
        all_points.append(x)
        iter_count += 1
    return x, iter_count, all_points
