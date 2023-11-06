# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:31:38 2021

@author: malga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

numEpocas = 1000      # Número de épocas.
q = 143               # Número de padrões. (Assuming you have 100 data points)

eta = 0.03            # Taxa de aprendizado (é interessante alterar para avaliar o comportamento)
m = 6                 # Número de neurônios na camada de entrada (infoavail, housecost, schoolquality, policetrust)
N = 8                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída. (Iris-setosa or Iris-versicolor)

# Carrega os dados de treinamento
df = pd.read_csv('happydata.csv')

# Extraia os dados das colunas
infoavail = df['infoavail'].to_numpy()
housecost = df['housecost'].to_numpy()
schoolquality = df['schoolquality'].to_numpy()
policetrust = df['policetrust'].to_numpy()
streetquality = df['streetquality'].to_numpy()
events = df['events'].to_numpy()

# Converta a coluna 'happy' em valores numéricos (0 para Iris-setosa e 1 para Iris-versicolor)
happy = df['happy'].to_numpy()

# Inicia aleatoriamente as matrizes de pesos.
W1 = np.random.random((N, m + 1))  # Dimensões da Matriz de entrada
W2 = np.random.random((L, N + 1))  # Dimensões da Matriz de saída

# Array para amazernar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas)  # Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# Bias
bias = 1

# Entrada do Perceptron.
X = np.vstack((infoavail, housecost, schoolquality, policetrust, streetquality, events))  # Concatenação dos quatro vetores

# ===============================================================
# TREINAMENTO.
# ===============================================================

for i in range(numEpocas):  # Repete o número de épocas determinado, neste caso 30000
    for j in range(q):  # Repete o número de dados existentes (neste exemplo, 100)

        # Insere o bias no vetor de entrada (apresentação do padrão da rede)
        Xb = np.hstack((bias, X[:, j]))

        # Saída da Camada Escondida.
        o1 = np.tanh(W1.dot(Xb))

        # Incluindo o bias. Saída da camada escondida é a entrada da camada de saída.
        o1b = np.insert(o1, 0, bias)

        # Neural network output
        Y = np.tanh(W2.dot(o1b))

        e = happy[j] - Y  # Equação (5).

        # Erro Total.
        E[j] = (e.transpose().dot(e)) / 2  # Equação de erro quadrática.

        # Error backpropagation.
        # Cálculo do gradiente na camada de saída.
        delta2 = np.diag(e).dot((1 - Y * Y))  # Eq. (6)
        vdelta2 = (W2.transpose()).dot(delta2)  # Eq. (7)
        delta1 = np.diag(1 - o1b * o1b).dot(vdelta2)  # Eq. (8)

        # Atualização dos pesos.
        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2, o1b))

    # Calculo da média dos erros
    Etm[i] = E.mean()

plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.plot(Etm)
plt.show()


# ===============================================================
# TESTE DA REDE.
# ===============================================================

Error_Test = np.zeros(q)

for i in range(q):
    # Insere o bias no vetor de entrada.
    Xb = np.hstack((bias, X[:, i]))

    # Saída da Camada Escondida.
    o1 = np.tanh(W1.dot(Xb))

    # Incluindo o bias. Saída da camada escondida é a entrada da camada de saída.
    o1b = np.insert(o1, 0, bias)

    # Neural network output
    Y = np.where(W2.dot(o1b) <= 0, 1, 0)

    # Extract a single element from the happy array
    target = happy[i]

    Error_Test[i] = target - Y[0]


print(f'Error_Test:\n{Error_Test}')
print(f'Round:\n{np.round(Error_Test) - happy[:q]}')
