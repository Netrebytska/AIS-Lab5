import numpy as np
import neurolab as nl
import numpy.random as rand
import pylab as pl

skv = 0.05
centr = np.array([[0.2, 0.2], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5]])
rand_norm = skv * rand.randn(100, 4, 2)
inp = np.array([centr + r for r in rand_norm])
inp.shape = (100 * 4, 2)
rand.shuffle(inp)

net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 4)
error = net.train(inp, epochs=200, show=20)

pl.figure()
pl.plot(error)
pl.title('Error Over Epochs')
pl.xlabel('Epoch number')
pl.ylabel('Error (default MAE)')
pl.savefig('error_over_epochs.png')

w = net.layers[0].np['w']
pl.figure()
pl.plot(inp[:, 0], inp[:, 1], '.',
        centr[:, 0], centr[:, 1], 'yv',
        w[:, 0], w[:, 1], 'p')
pl.title('Train Samples and Centers')
pl.legend(['Train samples', 'Real centers', 'Train centers'])
pl.savefig('train_samples_and_centers.png')

pl.show()
