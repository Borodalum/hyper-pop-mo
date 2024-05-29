import time

import numpy as np


def calc_time(func, arguments):
    start = time.time()
    x, iterations, all_points = func(*arguments)
    end = time.time() - start
    return x, iterations, end, np.array(all_points)
