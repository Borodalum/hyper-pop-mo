import numpy as np

from calc_time_hyperparam import calc_time


def newton_with_wolfe_search(f, grad_f, hessian_f, x0, epsilon=1e-8, max_iter=10000):
    return calc_time(minimize, (f, grad_f, hessian_f, x0, epsilon, max_iter))


def wolfe_line_search(f, grad_f, x, p, c1=0.01, c2=0.9, max_iter=1000):
    alpha = 1.0
    for i in range(max_iter):
        if f(x + alpha * p) <= f(x) + c1 * alpha * np.dot(grad_f(x), p) and np.dot(grad_f(x + alpha * p),
                                                                                   p) >= c2 * np.dot(grad_f(x), p):
            return alpha
        alpha *= 0.5
    return None


def minimize(f, grad_f, hessian_f, x0, epsilon, max_iter, line_search_func=wolfe_line_search):
    x = np.array(x0, dtype=float)  # Ensure x is a float array
    all_points = [x]
    iter_count = 0
    while iter_count < max_iter:
        grad = grad_f(x)
        hess = hessian_f(x)
        delta_x = np.linalg.solve(hess, -grad)
        # learning_rate = backtracking_line_search(f, grad, x, delta_x)
        learning_rate = line_search_func(f, grad_f, x, delta_x)
        x += learning_rate * delta_x
        if np.linalg.norm(grad) < epsilon:
            break
        iter_count += 1
        all_points.append(x)
    return x, iter_count, all_points
