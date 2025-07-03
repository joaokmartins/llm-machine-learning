# Dados de entrada (X) e saída (y) - relação simples y = 2x + 1
X = [1, 2, 3, 4, 5]  # Entradas
y = [3, 5, 7, 9, 11]  # Saídas esperadas (2x + 1)

# Inicialização dos parâmetros (coeficiente w e intercepto b)
w = 0  # Peso (slope)
b = 0  # Intercepto (bias)
learning_rate = 0.01  # Taxa de aprendizado
epochs = 1000  # Número de iterações de treinamento

# Treinamento do modelo
for epoch in range(epochs):
    total_error = 0
    for i in range(len(X)):
        # Predição usando o modelo atual
        y_pred = w * X[i] + b
        
        # Calcula o erro (diferença entre predição e saída esperada)
        error = y_pred - y[i]
        total_error += error**2
        
        # Ajusta os parâmetros w e b com base no erro (gradiente descendente simples)
        w -= learning_rate * error * X[i]  # Atualiza o peso
        b -= learning_rate * error         # Atualiza o intercepto
    
    # Exibir o erro total a cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Erro total = {total_error:.4f}")

# Modelo treinado
print("\n=== Modelo treinado ===")
print(f"Peso (w): {w:.4f}")
print(f"Intercepto (b): {b:.4f}")

# Testando o modelo treinado
teste_X = 6
resultado = w * teste_X + b
print(f"\nPara entrada X = {teste_X}, a predição do modelo é y = {resultado:.2f}")