import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

data = np.loadtxt('data_perceptron.txt')

X = data[:, :2]
y = data[:, 2:]

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Вхідні дані')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.show()

min_val = X.min(axis=0)
max_val = X.max(axis=0)

net = nl.net.newp([min_val, max_val], 1)

error = net.train(X, y, epochs=100, show=20, lr=0.01)

plt.figure(figsize=(10, 6))
plt.plot(error)
plt.title('Графік просування процесу навчання')
plt.xlabel('Епохи')
plt.ylabel('Середня квадратична помилка')
plt.show()

test_data = np.array([[0.5, 0.5], [-0.5, -0.5]])
print("Передбачення для тестових даних:", net.sim(test_data))
