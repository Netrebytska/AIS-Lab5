#Варіант 16
import numpy as np
import neurolab as nl
import numpy.random as rand
import matplotlib.pyplot as plt

skv = 0.05
centr = np.array([[0.2, 0.2], [0.4, 0.4], [0.3, 0.3], [0.3, 0.6], [0.5, 0.7]])
rand_norm = skv * rand.randn(100, 5, 2)
inp = np.array([centr + r for r in rand_norm])
inp.shape = (100 * 5, 2)
rand.shuffle(inp)

net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 5)
error = net.train(inp, epochs=200, show=20)

# Графік 1: Зміна помилки від кількості епох
plt.figure()
plt.plot(error)
plt.title('Зміна помилки від кількості епох')
plt.xlabel('Номер епохи')
plt.ylabel('Помилка (MAE)')
plt.savefig('error_plot.png')

# Графік 2: Вхідні дані, реальні центри та центри після тренування
w = net.layers[0].np['w']
plt.figure()
plt.plot(inp[:, 0], inp[:, 1], '.', label='Навчальні зразки')
plt.plot(centr[:, 0], centr[:, 1], 'yv', label='Реальні центри')
plt.plot(w[:, 0], w[:, 1], 'p', label='Центри після тренування')
plt.legend()
plt.title('Класифікація задачі')
plt.savefig('cluster_centers.png')

plt.show()
