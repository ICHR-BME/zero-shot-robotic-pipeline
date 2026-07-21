# 1. Mete aquí tus datos del Punto Bajo
z_robot_1 = 179.1  # Reemplaza con tu Z real en la mesa
y_pixel_1 = 237    # Reemplaza con el Píxel Y de la mesa

# 2. Mete aquí tus datos del Punto Alto
z_robot_2 = 254.7  # Reemplaza con tu Z elevada
y_pixel_2 =  86   # Reemplaza con el Píxel Y elevado

# La matemática (Calculando pendiente y offset)
m = (z_robot_2 - z_robot_1) / (y_pixel_2 - y_pixel_1)
b = z_robot_1 - (m * y_pixel_1)

print("\n¡Copia y pega esta línea exacta en tu mainmatrix5.py (Línea 501):")
print(f"cz = float(np.clip({m:.4f} * cy_side + {b:.2f}, 40.0, 220.0))")