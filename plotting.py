import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(f, x_opt, y_opt, x_range=(-7.5, 7.5), y_range=(-5, 5)):
    # Создание 3D графика
    fig = plt.figure("Градиентный спуск")
    X = np.linspace(*x_range, 40)
    Y = np.linspace(*y_range, 40)
    X, Y = np.meshgrid(X, Y)
    Z = f([X, Y])

    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3d.set_title("График функции")
    ax_3d.plot_wireframe(X, Y, Z)

    x_opt = x_opt[:50]
    y_opt = y_opt[:50]

    z_opt = np.array([f([x_opt[i], y_opt[i]]) for i in range(len(x_opt))])
    ax_3d.scatter3D(x_opt, y_opt, z_opt, c='red', marker='.', alpha=1, s=50)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("Линии уровня")
    ax.contour(X, Y, Z, len(y_opt) if len(y_opt) < 25 else 25)

    # Соединяем точки стрелками по порядку
    for i in range(len(x_opt) - 1):
        ax.arrow(x_opt[i, 0], y_opt[i, 0], x_opt[i + 1, 0] - x_opt[i, 0], y_opt[i + 1, 0] - y_opt[i, 0], head_width=0.1, head_length=0.1)

    ax.plot(x_opt, y_opt, '.', c='red')
    plt.show()
