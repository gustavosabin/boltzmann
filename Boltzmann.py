import math

import matplotlib.pyplot as plt
import numpy as np

from src.PerceptronClass import PerceptronClass

#dados

x = np.array([
    [1, 0, 1, 1, 1, 0 ,0],
    [1, 0, 1, 1, 1, 0 ,0],
    [1, 1, 1, 1, 0, 0 ,0],
    [1, 0, 1, 1, 0, 0 ,1],
    [1, 0, 1, 1, 1, 0 ,0],
    [1, 1, 0, 0, 0, 1 ,1]
])

w = np.array([
    [0, 0.0, 0.0],
    [0, 0.1, 0.2],
    [0, 0.3, 0.4],
    [0, 0.5, 0.6],
    [0, 0.7, 0.8],
    [0, 0.9, 1.0],
    [0, 0.1, 0.2]
])

txAprendizagem = 0.1
maxEpocas = 5000
precisao = 0.000000000005

h1 = PerceptronClass(x, w, txAprendizagem)

vetErro = np.zeros(maxEpocas,)

erroAnt = 0
erroAtual = 0
for epoca in range(maxEpocas):
    h1.epoca()
    erroAnt = erroAtual
    erroAtual = h1.erro
    vetErro[epoca] = erroAtual
    derro = abs(erroAtual - erroAnt)
    print('erroAtual: %.10f  -  erroAnt: %f      derro: %f' % (erroAtual, erroAnt, derro))
    if (derro <= precisao):
        print('Aprendizagem convergiu em %d épocas com precisão de %f' % (epoca, precisao))
        break

if (derro > precisao):
    print('Aprendizagem não conseguiu convergir em %d épocas para a precisão determinada' % epoca)


print('w:')
print(h1.w)
print()

plt.plot(vetErro)
plt.show()