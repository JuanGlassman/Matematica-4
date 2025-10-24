import pandas as pd
import statsmodels.api as sm

# ---------------------------
# 1. Cargar el CSV
# ---------------------------
df = pd.read_csv("dataset_Facebook.csv", sep=";")

# ---------------------------
# 2. Seleccionar variables
# ---------------------------
cols = ["Total Interactions", "like", "comment", "share", "Category", "Post Hour", "Post Month"]
df = df[cols].copy()

# ---------------------------
# 3. Limpiar datos
# ---------------------------
# Convertir a numérico (reemplaza texto no convertible por NaN)
df = df.apply(pd.to_numeric, errors="coerce")

# Eliminar filas con NaN (faltantes)
df = df.dropna()

# ---------------------------
# 4. Definir X e Y
# ---------------------------
y = df["Total Interactions"]
X = df[["like", "comment", "share", "Category", "Post Hour", "Post Month"]]

# ---------------------------
# 5. Agregar constante
# ---------------------------
X = sm.add_constant(X)

# ---------------------------
# 6. Ajustar modelo MCO
# ---------------------------
model = sm.OLS(y, X).fit()

# ---------------------------
# 7. Mostrar resultados
# ---------------------------
print("\nResumen del modelo:")
print(model.summary())

print("\nCoeficientes del modelo (Mínimos Cuadrados Ordinarios):\n")
for variable, coef in model.params.items():
    print(f"{variable:12s} -> {coef:.6f}")

