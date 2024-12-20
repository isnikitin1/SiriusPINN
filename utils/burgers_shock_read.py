#чтение файла численного решения
import scipy.io
import numpy as np

data = scipy.io.loadmat('./burgers_shock.mat')
t = data['t'].flatten()[:,None] #100*1
x = data['x'].flatten()[:,None] #256*1
Exact = np.real(data['usol']).T #100*256
X,T = np.meshgrid(x, t)
