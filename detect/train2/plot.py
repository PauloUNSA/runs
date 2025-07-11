import pandas as pd
import matplotlib.pyplot as plt

# Cargar resultados del entrenamiento
df = pd.read_csv("results.csv")

# Eliminar espacios innecesarios en nombres de columnas (por si acaso)
df.columns = df.columns.str.strip()

# Crear una figura con múltiples curvas
plt.figure(figsize=(12, 6))

# ➤ Pérdidas de entrenamiento
plt.plot(df['train/box_loss'], label='Box Loss (train)', linestyle='-', marker='o')
plt.plot(df['train/cls_loss'], label='Class Loss (train)', linestyle='-', marker='o')
plt.plot(df['train/dfl_loss'], label='DFL Loss (train)', linestyle='-', marker='o')

# ➤ Métricas de validación
plt.plot(df['metrics/mAP50(B)'], label='mAP@50 (val)', linestyle='--', marker='x')
plt.plot(df['metrics/mAP50-95(B)'], label='mAP@50-95 (val)', linestyle='--', marker='x')
plt.plot(df['metrics/precision(B)'], label='Precision (val)', linestyle='--')
plt.plot(df['metrics/recall(B)'], label='Recall (val)', linestyle='--')

# ➤ Configuración del gráfico
plt.xlabel('Epoch')
plt.ylabel('Valor')
plt.title('Curvas de Aprendizaje YOLOv12s')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()