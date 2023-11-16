import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib import colors
df = pd.DataFrame(pd.read_csv('happydata.csv')).dropna()

numEpocas = 1     # Número de épocas.
q = len(df)                # Número de padrões.

eta = 0.1         # Taxa de aprendizado ( é interessante alterar para avaliar o comportamento)
m =  10                # Número de neurônios na camada de entrada (Age, Gender... Family hist, Stress levels)
N = 20                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída. (0 = No Stroke E 1 = Stroke)


infoavail = df["infoavail"].to_numpy()
housecost = df["housecost"].to_numpy()
schoolquality = df["schoolquality"].to_numpy()
policetrust = df["policetrust"].to_numpy()
streetquality = df["streetquality"].to_numpy()
events = df["events"].to_numpy()

y = df['happy'].to_numpy()

W1 = np.random.random((N, m + 1))
W2 = np.random.random((N, N + 1))
W3 = np.random.random((N, N + 1))
W4 = np.random.random((L, N + 1))

E = np.zeros(q)
Etm = np.zeros(numEpocas)

bias = 1

X = np.vstack((infoavail, housecost, schoolquality, policetrust, streetquality, events))

for i in range(numEpocas):
    for j in range(q):

        Xb = np.hstack((bias, X[:,j]))
        o1 = np.tanh(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)
        o2 = np.tanh(W2.dot(o1b))
        o2b = np.insert(o2, 0, bias)
        o3 = np.tanh(W3.dot(o2b))
        o3b = np.insert(o3, 0, bias)

        Y = np.tanh(W4.DOT(o3b))

        e = y[j] - Y

        E[j] = (e.transpose().dot(e))/2

        print(f'i = {str(i)} E = {str(E)}')

        delta4 = np.diag(e).dot(1 - Y * Y)
        vdelta4 = (W4.transpose()).dot(delta4)
        delta3 = np.diag(1 - o3b * o3b).dot(vdelta4)

        vdelta3 = (W3.transpose()).dot(delta3[1:])
        delta2 = np.diag(1 - o2b * o2b).dot(vdelta3)

        vdelta2 = (W2.transpose()).dot(delta2[1:])
        delta1 = np.diag(1 - o1b * o1b).dot(vdelta2)

        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2[1:], o2b))
        W3 = W3 + eta * (np.outer(delta3[1:], o2b))
        W4 = W4 + eta * (np.outer(delta4, o3b))

        Etm[i] = E.mean()

    plt.xlabel("Épocas")
    plt.ylabel("Erro Médio")
    plt.plot(Etm, color='b')
    plt.plot(Etm)
    plt.show()

    escolha = int(input('Deseja salvat o treino? 1 = sim\n'))
    if escolha == 1:
        joblib.dump(W1, 'W1_pesos.pkl')
        joblib.dump(W2, 'W2_pesos.pkl')
        joblib.dump(W3, 'W3_pesos.pkl')
        joblib.dump(W4, 'W4_pesos.pkl')
    
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
        o2 = np.tanh(W2.dot(o1b))
        o2b = np.insert(o2, 0, bias)
        o3 = np.tanh(W3.dot(o2b))
        o3b = np.insert(o3, 0, bias)

        # Neural network output
        Y = np.tanh(W2.dot(o1b))            # Equações (3) e (4) juntas.
        print(Y)

        

        
        Error_Test[i] = y[i] - (Y)
    print(Error_Test)
    print(np.round(Error_Test) - y) #aqui se ela acertou todas o vetor tem que estar zerado

informacoes = int(input("infoavail: "))
customoradia = int(input("housecost:  "))
escola = int(input("schoolquality: "))
policia = int(input("policetrust: "))
rua = float(input("streetquality: "))
evento = int(input("events: "))


input_usuario = []
input_usuario.append(informacoes, customoradia, escola, policia, rua, evento)
print(input_usuario)

