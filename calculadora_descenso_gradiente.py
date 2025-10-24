import numpy as np
import pandas as pd

# ---------------------------
# CARGA DE DATOS DESDE CSV
# ---------------------------
df = pd.read_csv("dataset_Facebook.csv", sep=";")

# Mostrar las columnas disponibles
print("Columnas disponibles:\n", df.columns, "\n")

# ---------------------------
# SELECCIÓN Y LIMPIEZA DE DATOS
# ---------------------------
# Eliminar filas con valores faltantes en las columnas necesarias
df = df.dropna(subset=["Total Interactions", "like", "comment", "share", "Category", "Post Hour", "Post Month"])

# Definir variables dependiente e independientes
Y = df["Total Interactions"].to_numpy(dtype=float)
X = df[["like", "comment", "share", "Category", "Post Hour", "Post Month"]].to_numpy(dtype=float)

# ---------------------------
# NORMALIZACIÓN (para mejorar la convergencia)
# ---------------------------
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Agregar columna de 1s para el intercepto β0
m = len(Y)
X = np.hstack((np.ones((m, 1)), X))

# ---------------------------
# PARÁMETROS
# ---------------------------
e = 1e-6     # tolerancia (ε)
n = 0.01     # tasa de aprendizaje (η)
betas = np.zeros(X.shape[1])  # β iniciales
max_iter = 100000
i = 1

# ---------------------------
# FUNCIONES
# ---------------------------
def f(betas):
    """Función de costo: Error cuadrático medio (MSE)"""
    Y_pred = X.dot(betas)
    return np.mean((Y - Y_pred) ** 2)

def gradiente(betas):
    """Gradiente del MSE respecto a β"""
    Y_pred = X.dot(betas)
    error = Y_pred - Y
    return (2/m) * X.T.dot(error)

def corte(b0, b1):
    """Condición de corte: cambio pequeño en la función de costo"""
    return abs(f(b1) - f(b0)) < e

# ---------------------------
# DESCENSO DEL GRADIENTE
# ---------------------------
print("Iteraciones paso a paso:\n")

while True:
    grad = gradiente(betas)
    betas_nuevas = betas - n * grad
    f_old = f(betas)
    f_new = f(betas_nuevas)
    diff = abs(f_new - f_old)
    
    if i % 1000 == 0:
        print(f"Iteración {i}: costo = {f_new:.6f}, Δ={diff:.6e}")

    if corte(betas, betas_nuevas) or i >= max_iter:
        betas = betas_nuevas
        break
    
    betas = betas_nuevas
    i += 1

print(f"\n✅ Convergencia alcanzada en {i} iteraciones.")

# --- IMPRIMIR LA ECUACIÓN ---
print("\n" + "="*40)
print("ECUACIÓN DE REGRESIÓN (Descenso del Gradiente):")

ecuacion_str = f"Ŷ = {betas[0]:.4f}"
for j in range(1, len(betas)):
    coef = betas[j]
    signo = "+" if coef >= 0 else "-"
    valor_abs = abs(coef)
    ecuacion_str += f" {signo} {valor_abs:.4f}*X{j}"

print(ecuacion_str)
print("="*40)
