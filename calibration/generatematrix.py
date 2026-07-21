import cv2
import numpy as np

# 1. Pon aquí los 4 píxeles de tu cámara [X, Y]
# Orden: Superior-Izq, Superior-Der, Inferior-Izq, Inferior-Der
puntos_pixel = np.array([
    [164, 59],  # Esquina 1 (Píxeles)
    [406, 59],  # Esquina 2 (Píxeles)
    [158, 326],  # Esquina 3 (Píxeles)
    [415, 328]   # Esquina 4 (Píxeles)
], dtype=np.float32)

# 2. Coordenadas perfectas (Left/Right fixed AND Up/Down inverted)
puntos_robot = np.array([
    [205.9, 50.9],   # (Era Esq 3) Ahora asignado a Esq 1 (Sup-Izq)
    [203.9, 211.2],  # (Era Esq 4) Ahora asignado a Esq 2 (Sup-Der)
    [379.5, 50.9],   # (Era Esq 1) Ahora asignado a Esq 3 (Inf-Izq)
    [372.8, 226.6]   # (Era Esq 2) Ahora asignado a Esq 4 (Inf-Der)
], dtype=np.float32)

# 3. La Magia Matemática: Calcular la Matriz de Homografía (3x3)
matriz_homografia, estado = cv2.findHomography(puntos_pixel, puntos_robot)

print("¡Matriz calculada con éxito!\n")
print(matriz_homografia)

# 4. Guardar la matriz en un archivo para que tu código principal la use
np.save("matriz_vision_xarm.npy", matriz_homografia)
print("\nMatriz guardada como 'matriz_vision_xarm.npy'")