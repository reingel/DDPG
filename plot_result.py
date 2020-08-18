import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

a = np.loadtxt('result.txt', delimiter=',', skiprows=1)
episode = a[:,0]
step = a[:,1]
R = a[:,2]
noise = a[:,3]

plt.figure(100)
plt.subplot(3,1,1)
plt.plot(episode, step)
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(episode, R)
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(episode, noise)
plt.grid(True)
plt.show()