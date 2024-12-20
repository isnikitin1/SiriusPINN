#чтение файла численного решения
import scipy.io
import numpy as np

def load(path='./burgers_shock.mat'):
    data = scipy.io.loadmat(path)
    t = data['t'].flatten()[:,None] #100*1
    x = data['x'].flatten()[:,None] #256*1
    U = np.real(data['usol']).T #100*256
    X,T = np.meshgrid(x, t)
    return X, T, U
