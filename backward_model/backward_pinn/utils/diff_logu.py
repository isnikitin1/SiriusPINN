import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm

def diff_logU(U,Exact):
    R = [[0]*256 for _ in range(100)]
    for i in range(100):
        for j in range(256):
            R[i][j] = abs(Exact[i][j] - U[i][j])
    fig, ax = plt.subplots()
    gr = ax.imshow(R, cmap='cool', extent=(-1, 1, -1, 1), norm=LogNorm())
    plt.title("")
    plt.xlabel('X')
    plt.ylabel('T')
    axins = inset_axes(ax, width="5%", height="100%", loc='upper left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    c = plt.colorbar(gr, cax=axins)
    plt.title(r'$\Delta U$')
    plt.show()
