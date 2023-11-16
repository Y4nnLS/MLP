# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:31:38 2021

@author: malga
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.DataFrame(pd.read_csv('happydata.csv')).dropna()

# Definindo as variáveis de entrada (X) e saída (y)
infoavail = df['infoavail'].to_numpy()
housecost = df['housecost'].to_numpy()
schoolquality = df['schoolquality'].to_numpy()
policetrust = df['policetrust'].to_numpy()
streetquality = df['streetquality'].to_numpy()
events = df['events'].to_numpy()

# Extrai os dados da coluna 'happy' que é a saida esperada
happy = df['happy'].to_numpy()
d = happy
# d = np.array_split(happy, 3)[0]
# print(happy)
print(d)

numEpocas = 400      # Número de épocas.
q = len(df)                # Número de padrões.

eta = 0.01            # Taxa de aprendizado ( é interessante alterar para avaliar o comportamento)
m = 6                 # Número de neurônios na camada de entrada (peso e PH)
N = 8                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída. (-1 = Maçã E 1 = Laranja)

# Carrega os dados de treinamento

# Inicia aleatoriamente as matrizes de pesos.
# Inicializando com m, N e L nos dá a liberdade de diferentes arquiteturas (só alterando as linhas 17,18 e 19)
W1 = np.random.random((N, m + 1)) #dimensões da Matriz de entrada
W2 = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
W3 = np.random.random((N, N + 1)) #dimensões da Matriz de entrada
W4 = np.random.random((L, N + 1)) #dimensões da Matriz de saída

# Array para amazernar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas) #Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# bias
bias = 1


# Função de ativação Sigmóide
def sigmoid(X):
    return 1/(1+np.exp(-X))
# Entrada do Perceptron.
X = np.vstack((infoavail, housecost, schoolquality, policetrust, streetquality, events ))   # concatenação dos dois vetores

# ===============================================================
# TREINAMENTO.
# ===============================================================
        
for i in range(numEpocas): #repete o numero de vezes terminado, no caso 20
    for j in range(q): #repete o numero de "dados" existentes (nesse exemplo 13)
        
        # Insere o bias no vetor de entrada (apresentação do padrão da rede)
        Xb = np.hstack((bias, X[:,j])) #empilhamos pelo hstack junto ao bias e ficamos 
                                       #com unico vetor [bias peso PH]

        # Saída da Camada Escondida.
        o1  = sigmoid(W1.dot(Xb)) 
        o1b = np.insert(o1, 0, bias)
        o2  = sigmoid(W2.dot(o1b)) 
        o2b = np.insert(o2, 0, bias)          
        o3  = sigmoid(W2.dot(o2b)) 
        o3b = np.insert(o3, 0, bias)

        # Neural network output
        Y = np.tanh(W4.dot(o1b))            # Equações (3) e (4) juntas.
                                            #Resulta na saída da rede neural
        
        # e = d[j] - Y                        # Equação (5).
        e = d[j % len(d)] - Y  # Equação (5).

        # Erro Total.
        E[j] = (e.transpose().dot(e))/2     # Equação de erro quadrática.
        
        # Imprime o número da época e o Erro Total.
        # print('i = ' + str(i) + '   E = ' + str(E))
   
        # Error backpropagation.   
        # Cálculo do gradiente na camada de saída.
        delta2 = np.diag(e).dot((1 - Y*Y))          # Eq. (6)
        # vdelta2 = (W2.transpose()).dot(delta2.reshape(-1, 1))  # Eq. (7)
        vdelta2 = (W4.transpose()).dot(delta2.reshape(-1, 1))  # Eq. (7)

        deltaU = np.diag(1 - o3b*o3b).dot(vdelta2)  # Eq. (8)
        vdeltaU = (W2.transpose()).dot(deltaU[1:])  # Eq. (9)
        deltaX = np.diag(1 - o2b*o2b).dot(vdeltaU)  # Eq. (10)
        vdeltaX = (W3.transpose()).dot(deltaX[1:])  # Eq. (11)
        delta1 = np.diag(1 - o1b*o1b).dot(vdeltaX)  # Eq. (12)

        # Atualização dos pesos.
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(deltaU[1:], o3b))
        W3 = W3 + eta*(np.outer(deltaX[1:], o2b))
        W4 = W4 + eta*(np.outer(delta2, o1b))
    
    #Calculo da média dos erros
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
    Xb = np.hstack((bias, X[:,i]))

    # Saída da Camada Escondida.
    o1 = np.tanh(W1.dot(Xb))            # Equações (1) e (2) juntas.      
    #print(o1)
    
    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    o1b = np.insert(o1, 0, bias)
    o2  = sigmoid(W2.dot(o1b)) 
    o2b = np.insert(o2, 0, bias)          
    o3  = sigmoid(W2.dot(o2b)) 
    o3b = np.insert(o3, 0, bias)

    # Neural network output
    Y = np.tanh(W2.dot(o1b))            # Equações (3) e (4) juntas.
    print(Y)
    
    Error_Test[i] = d[i] - Y[0]
    print(d[i] - Y[0])
    # for i, value in enumerate(Y):
    #     Error_Test[i] = d[i] - value


erros = 0

for i in range(len(Error_Test)):
    if np.round(Error_Test[i]) != 1:
        erros += 1    
print(np.round(Error_Test))
print('erros: ', erros)
print("Erros:\n" + str(np.round(Error_Test)))
print("Percentual: {:.2f}%".format((erros * 100)/q))