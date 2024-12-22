import matplotlib.pyplot as plt
from   utils import grcolor

#график при разных T
def plot_T(U, X):
    plt.xlabel('X')
    plt.ylabel('U')
    for i in range(0, 100, 10):
        plt.plot(X[i], U[i], label="T="+str(i/100), color=grcolor.grcolor((i+1)//10))
    plt.legend(loc='upper right')
    plt.title('')
    plt.show()
