import sys
sys.path.insert(0, r"C:/Modelo_Tesis/Modulo_I")

from config import RUTA_ICENTIA, RUTA_MITBIH
from modulo_1_clasificador.lectura_senal import cargar_dataset_balanceado

X, y = cargar_dataset_balanceado(
    [str(RUTA_ICENTIA), str(RUTA_MITBIH)],
    n_por_clase=10
)

import numpy as np
print(f"\nTotal señales cargadas : {len(X)}")
print(f"Clases detectadas      : {sorted(set(y))}")
print(f"Longitudes unicas      : {sorted(set(len(s) for s in X))}")
print("OK — listo para entrenamiento")