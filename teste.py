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
q = 100               # Número de padrões. (Assuming you have 100 data points)

eta = 0.01            # Taxa de aprendizado (é interessante alterar para avaliar o comportamento)
m = 4                 # Número de neurônios na camada de entrada (SepalLength, SepalWidth, PetalLength, PetalWidth)
N = 8                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída. (Iris-setosa or Iris-versicolor)

# Carrega os dados de treinamento
df = pd.read_csv('Iris.csv')

# Extraia os dados das colunas
SepalLength = df['SepalLength'].to_numpy()
SepalWidth = df['SepalWidth'].to_numpy()
PetalLength = df['PetalLength'].to_numpy()
PetalWidth = df['PetalWidth'].to_numpy()

# Converta a coluna 'Species' em valores numéricos (0 para Iris-setosa e 1 para Iris-versicolor)
Species = df['Species'].apply(lambda x: 0 if x == 'Iris-setosa' else 1).to_numpy()

# Inicia aleatoriamente as matrizes de pesos.
W1 = np.random.random((N, m + 1))  # Dimensões da Matriz de entrada
W2 = np.random.random((L, N + 1))  # Dimensões da Matriz de saída

# Array para amazernar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas)  # Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# Bias
bias = 1

# Entrada do Perceptron.
X = np.vstack((SepalLength, SepalWidth, PetalLength, PetalWidth))  # Concatenação dos quatro vetores

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

        e = Species[j] - Y  # Equação (5).

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

# ... (previous code remains the same)

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
    Y = np.tanh(W2.dot(o1b))

    # Extract a single element from the Species array
    target = Species[i]

    Error_Test[i] = target - Y[0]

print(Error_Test)
print(abs(np.round(Error_Test) - Species[:q]))
