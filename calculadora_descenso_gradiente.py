import numpy as np
import pandas as pd

# CARGA DE DATOS
df = pd.read_csv("dataset_Facebook.csv", sep=";")

# Seleccionar columnas de interés y eliminar filas con NaN
cols = ["Total Interactions", "like", "comment", "share", "Category", "Post Hour", "Post Month"]
df = df[cols].dropna()

# Variables dependiente e independientes
Y = df["Total Interactions"].to_numpy(dtype=float)
X = df[["like", "comment", "share", "Category", "Post Hour", "Post Month"]].to_numpy(dtype=float)



# NORMALIZACIÓN
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Agregar columna de 1s para el intercepto
m = len(Y)
X_norm = np.hstack((np.ones((m, 1)), X_norm))


# PARÁMETROS DESCENSO DEL GRADIENTE
eta = 0.01       # tasa de aprendizaje
epsilon = 1e-6   # tolerancia
betas = np.zeros(X_norm.shape[1])
max_iter = 100000

# Función de costo (MSE)
def mse(betas):
    Y_pred = X_norm.dot(betas)
    return np.mean((Y - Y_pred) ** 2)

# Gradiente del MSE
def gradiente(betas):
    Y_pred = X_norm.dot(betas)
    error = Y_pred - Y
    return (2/m) * X_norm.T.dot(error)

# Condición de corte
def convergencia(b0, b1):
    return abs(mse(b1) - mse(b0)) < epsilon


# DESCENSO DEL GRADIENTE
i = 1
while True:
    grad = gradiente(betas)
    betas_nuevas = betas - eta * grad
    if i % 1000 == 0:
        print(f"Iteración {i}: costo = {mse(betas_nuevas):.6f}")
    if convergencia(betas, betas_nuevas) or i >= max_iter:
        betas = betas_nuevas
        break
    betas = betas_nuevas
    i += 1

print(f"\n✅ Convergencia alcanzada en {i} iteraciones.")


# DESNORMALIZAR COEFICIENTES
beta0_orig = betas[0] - np.sum((betas[1:] * X_mean) / X_std)
beta_orig = betas[1:] / X_std


# IMPRIMIR ECUACIÓN FINAL
print("\n" + "="*40)
print("ECUACIÓN DE REGRESIÓN (Descenso del Gradiente, escala original):")
ecuacion_str = f"Ŷ = {beta0_orig:.6f}"
for j in range(len(beta_orig)):
    signo = "+" if beta_orig[j] >= 0 else "-"
    valor_abs = abs(beta_orig[j])
    ecuacion_str += f" {signo} {valor_abs:.6f}*X{j+1}"
print(ecuacion_str)
print("="*40)
