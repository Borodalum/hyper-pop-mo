from scipy.optimize import minimize

from calc_time_hyperparam import calc_time


def newton_cg(f, grad_f, x0):
    return calc_time(newton_cg_method, (f, grad_f, x0))


def quasinewton(f, grad_f, x0):
    return calc_time(bfgs_method, (f, grad_f, x0))


def newton_cg_method(f, grad_f, x0):
    result = minimize(f, x0, method='Newton-CG', jac=grad_f)
    return result.x, result.nit, None


def bfgs_method(f, grad_f, x0):
    result = minimize(f, x0, method='BFGS', jac=grad_f)
    return result.x, result.nit, None
