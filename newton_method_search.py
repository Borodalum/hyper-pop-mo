import numpy as np

from calc_time_hyperparam import calc_time


def newton_with_search_learning_rate(f, grad_f, hessian_f, x0, epsilon=1e-8, max_iter=10000):
    return calc_time(minimize, (f, grad_f, hessian_f, x0, epsilon, max_iter))


def minimize(f, grad_f, hessian_f, x0, epsilon, max_iter):
    x = np.array(x0, dtype=float)  # Ensure x is a float array
    all_points = [x]
    iter_count = 0
    while iter_count < max_iter:
        grad = grad_f(x)
        hess = hessian_f(x)
        delta_x = np.linalg.solve(hess, -grad)
        learning_rate = backtracking_line_search(f, grad, x, delta_x)
        x += learning_rate * delta_x
        if np.linalg.norm(grad) < epsilon:
            break
        iter_count += 1
        all_points.append(x)
    return x, iter_count, all_points


def backtracking_line_search(f, grad_f, x, delta_x, alpha=0.4, beta=0.8):
    learning_rate = 1.0
    while f(x + learning_rate * delta_x) > f(x) + alpha * learning_rate * np.dot(grad_f, delta_x):
        learning_rate *= beta
    return learning_rate
