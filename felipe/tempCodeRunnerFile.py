igite a sua idade: "))
genero = int(input("Voce eh Homem(1) ou Mulher(-1)?:  "))
hipertensao = int(input("Voce ja teve hipertensao? (1)s (0)n: "))
doenca_coracao = int(input("Voce ja teve alguma doenca do coracao? (1)s (0)n: "))
nivel_glucose = float(input("Digite o seu nivel de glucose: "))
status_fumo = int(input("Voce nao fuma(1), fuma casualmente(2) ou fuma com frequencia(3)?: "))
frequencia_treino = int(input("O seu nivel de treino eh: baixo(1), medio(2) ou alto(3)?: "))
hist_avc = int(input("Voce ja teve um AVC previamente? (1)s (0)n: "))
hist_familiar_avc = int(input("Voce tem um historico familiar de AVC? (1)s (0)n"))
nivel_stress = float(input("Digite o seu nivel de stresse: "))

input_usuario = []
input_usuario.append((idade, genero, hipertensao, doenca_coracao, nivel_glucose, status_fumo, frequencia_treino, hist_avc, hist_familiar_avc, nivel_stress))
print(input_usuario)