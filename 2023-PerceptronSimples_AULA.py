# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:12:08 2021

@author: malga
"""

import numpy as np 

# Define o número de épocas da simulação e o número de atributos
numEpocas = 100
numAmostras = 7

# Atributos
peso = np.array([113, 122, 107, 98, 115, 120, 118])
pH = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2, 5.1 ])


# bias
bias = 1

# Entrada do Perceptron.
#vstack = A função é usada para empilhar a sequência de matrizes de entrada
#verticalmente para fazer uma única matriz
X = np.vstack((peso, pH))   # Matriz de 2 colunas Ou X = np.asarray([peso, pH])
Y = np.array([-1, 1, -1, -1, 1, 1, 1]) # Nossa saída


# Taxa de aprendizado.
eta = 0.1

# Define o vetor de pesos inicializamos com ZERO
W = np.zeros([1,3])           # (1 linha e 3 colunas) = Duas entradas + o bias !

# Array para amazernar os erros.
e = np.zeros(7)

"""
def funcaoAtivacao(valor):
    #A função de ativação Degrau Bipolar
    
    if valor < 0.0:
        return(-1)
    else: 
        return(1)
 """ 
#Parte principal envolvendo o treinamento
#-------------------------------------------------------------

for j in range(numEpocas):
    for k in range(numAmostras): # 6 vezes pois é nosso numero de amostras
        
     # Insere o bias no vetor de entrada.
     Xb = np.hstack((bias, X[:,k]))# O Hstack vai empilhar o bias em todas as linhas (:) indice K
     
     # Calcula o vetor campo induzido (multiplicação vetorial)
     #numpy.dot retorna o produto escalar dos vetores de entrada.
     V = np.dot(W, Xb)
     
     # Calcula a saída do perceptron.
     Yr = np.tanh(V) #recebe o valor do campo induzido
     
     # Calcula o erro: e = (Y - Yr)
     e[k] = Y[k] - Yr # saída que a gente conhece - a saída da rede
     
     # Treinando a rede.
     W = W + eta*e[k]*Xb #peso + a taxa de aprendizado*erro*a nossa entrada ajustada com o bias
     
     
#print(W)
#colocar dentro do for se desejar printar cada época de aprendizado
print("Vetor de errors (e) = " + str(e)) #transformamos o vetor numérico em um strig.

import matplotlib.pyplot as plt
plt.scatter(peso,pH)
plt.show()


#np.tanh(V)
#np.sign(V)