# импорт модулей
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import json
import imageio
import random


# цвета графика
def grcolor(c):
    C = ["#fd02ff", "#df20ff", "#bf40ff", "#9e61ff", "#7f80ff", "#5ea1ff", "#4bb4ff", "#2ed1ff", "#27d8ff", "#01feff"]
    return C[c]


def plot_points(x, u):
    plt.xlabel('X')
    plt.ylabel('U')
    plt.plot(x, u)
    plt.title('')
    plt.show()


# график при разных T
def plot_T(U, X):
    plt.xlabel('X')
    plt.ylabel('U')
    for i in range(0, 100, 10):
        plt.plot(X[i], U[i], label="T=" + str(i / 100), color=grcolor((i + 1) // 10))
    plt.legend(loc='upper right')
    plt.title('')
    plt.show()


# график U
def plot_U(U):
    fig, ax = plt.subplots()
    gr = ax.imshow(U, cmap='cool', extent=(-1, 1, 0, 1))
    plt.xlabel('X')
    plt.ylabel('T')
    axins = inset_axes(ax, width="5%", height="100%", loc='upper left', bbox_to_anchor=(1.03, 0., 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    plt.colorbar(gr, cax=axins)
    plt.title('U')
    plt.show()


# график ошибки U
def diff_U(U, Exact):
    R = [[0] * 256 for _ in range(100)]
    for i in range(100):
        for j in range(256):
            R[i][j] = abs(Exact[i][j] - U[i][j])
    fig, ax = plt.subplots()
    gr = ax.imshow(R, cmap='cool', extent=(-1, 1, 0, 1), vmin=0, vmax=1)
    plt.xlabel('X')
    plt.ylabel('T')
    axins = inset_axes(ax, width="5%", height="100%", loc='upper left', bbox_to_anchor=(1.03, 0., 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    plt.colorbar(gr, cax=axins)
    plt.title(r'$\Delta U$')
    plt.show()


# график log ошибки U
def diff_logU(U, Exact):
    R = [[0] * 256 for _ in range(100)]
    for i in range(100):
        for j in range(256):
            R[i][j] = abs(Exact[i][j] - U[i][j])
    fig, ax = plt.subplots()
    gr = ax.imshow(R, cmap='cool', extent=(-1, 1, 0, 1), norm=LogNorm())
    plt.title("")
    plt.xlabel('X')
    plt.ylabel('T')
    axins = inset_axes(ax, width="5%", height="100%", loc='upper left', bbox_to_anchor=(1.03, 0., 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0)
    c = plt.colorbar(gr, cax=axins)
    plt.title(r'$\Delta U$')
    plt.show()


# gif графика на разных разных T
def gif_grU(U, X):
    for i in range(100):
        plt.ylim(-1.1, 1.1)
        plt.xlabel('X')
        plt.ylabel('U')
        plt.title('T=' + str(round(float(t[i]), 2)))
        plt.plot(X[i], U[i], color="#000000")
        plt.savefig(str(i) + '.png')
        plt.show()
    with imageio.get_writer('animation.gif', mode='I') as writer:
        for i in range(100):
            writer.append_data(imageio.imread(str(i) + '.png'))


# gif по разным T + график численного решения
def gif_grU1(U, X, Exact):
    for i in range(100):
        plt.ylim(-1.1, 1.1)
        plt.xlabel('X')
        plt.ylabel('U')
        plt.title('T=' + str(round(float(t[i]), 2)))
        plt.plot(X[i], U[i], color="blue")
        plt.plot(X[i], Exact[i], "--", color="red")
        plt.savefig(str(i) + '.png')
        plt.show()
    with imageio.get_writer('animation.gif', mode='I') as writer:
        for i in range(100):
            writer.append_data(imageio.imread(str(i) + '.png'))
def dataset_plot(x,t,U):
    fig, ax = plt.subplots()
    plt.scatter(x, t, color = '#000000',marker = "x",clip_on=False)
    gr = ax.imshow(U,cmap='cool',extent=(-1, 1, -0,1))
    plt.xlabel('X')
    plt.ylabel('T')
    axins = inset_axes(ax, width="5%", height="100%", loc='upper left', bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    
    plt.colorbar(gr, cax=axins)
    plt.title('U')
    plt.show()
