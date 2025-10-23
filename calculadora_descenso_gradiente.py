import numpy as np

# ---------------------------
# DATOS
# ---------------------------
Y = np.array([7, 12, 15, 39, 45, 58, 62, 79, 92, 114], dtype=float)
X = np.array([
    [7, 0, 0, 1, 11, 10],
    [11, 0, 1, 1, 8, 10],
    [14, 0, 1, 1, 4, 6],
    [27, 1, 11, 2, 10, 12],
    [33, 2, 10, 3, 3, 6],
    [48, 1, 9, 2, 9, 11],
    [48, 0, 14, 3, 2, 4],
    [63, 2, 14, 3, 10, 10],
    [77, 0, 15, 1, 11, 12],
    [112, 4, 8, 1, 10, 9]
], dtype=float)

# Agregar columna de 1s para el intercepto β0
m = len(Y)
X = np.hstack((np.ones((m, 1)), X))

# ---------------------------
# PARÁMETROS
# ---------------------------
e = 0.0001    # tolerancia (ε)
n = 0.0001    # tasa de aprendizaje (η)
betas = np.zeros(X.shape[1])  # β iniciales
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
    
    print(f"Iteración {i}:")
    print(f"  Gradiente: {np.round(grad,4)}")
    print(f"  Betas anteriores: {np.round(betas,4)}")
    print(f"  Betas nuevas: {np.round(betas_nuevas,4)}")
    print(f"  Costo anterior: {f_old:.6f} | Costo nuevo: {f_new:.6f} | Δ={diff:.6f}\n")
    
    if corte(betas, betas_nuevas):
        print("\nSe cumple la condición de corte. El algoritmo se detiene.")
        betas = betas_nuevas
        break
    
    betas = betas_nuevas
    i += 1

# --- IMPRIMIR LA ECUACIÓN ---
print("\n" + "="*30)
print("ECUACIÓN DE REGRESIÓN (Descenso Gradiente):")

# Iniciar la ecuación con el intercepto (B0)
# Usamos :.4f para redondear a 4 decimales
ecuacion_str = f"Ŷ = {betas[0]:.4f}"

# Iterar sobre los otros coeficientes (B1 a B6)
for i in range(1, len(betas)):
    coef = betas[i]
    
    # Definir el signo (+ o -)
    signo = "+" if coef >= 0 else "-"
    
    # Obtener el valor absoluto del coeficiente
    valor_abs = abs(coef)
    
    # Agregar a la ecuación
    ecuacion_str += f" {signo} {valor_abs:.4f}*X{i}"

print(ecuacion_str)
print("="*30)




