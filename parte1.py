import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
# from google.colab import files

#subir archivo zip


folder = "TP4_imagenes"
y_sustrato = 130

#funciones
def procesar_imagen(img_path, y_sustrato):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None, None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    ys, xs = np.where(edges > 0)
    ys_contacto = y_sustrato - ys  #eje Y hacia arriba desde el sustrato
    return xs, ys_contacto, img, thresh

def calcular_centro(xs, ys_contacto):
    mask = ys_contacto >= 0
    xs_gota = xs[mask]
    ys_gota = ys_contacto[mask]
    if xs_gota.size == 0:
        return 0.0, 0.0
    return np.median(xs_gota), np.median(ys_gota)

def graficar_tres_paneles(xs, ys_contacto, centro_x, centro_y, img, thresh, frame_num):
    height, width = img.shape
    fig, axs = plt.subplots(1, 3, figsize=(18,6))

    #contorno con centro y sustrato (ahora primero)
    axs[0].scatter(xs, ys_contacto, s=1, label="Contorno", color='purple')
    axs[0].axhline(y=0, color='red', linestyle='--', label="Sustrato Y=0")
    axs[0].scatter(centro_x, centro_y, color='gold', s=50, label="Centro gota")
    axs[0].set_xlim(0, width)
    axs[0].set_ylim(ys_contacto.min()-10, ys_contacto.max()+10)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlabel("X", color='teal')
    axs[0].set_ylabel("Y respecto al sustrato (hacia arriba)", color='teal')
    axs[0].set_title(f"Contorno - Frame {frame_num}", color='purple')
    axs[0].legend()

    #imagen segmentada (ahora segundo)
    axs[1].imshow(thresh, cmap='gray')
    axs[1].set_title("Segmentada", color='darkgreen')
    axs[1].axis('off')

    #imagen original (ahora tercero)
    axs[2].imshow(img, cmap='gray')
    axs[2].set_title("Original", color='navy')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()


import random

imagenes = natsorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
print(f"Cantidad de imágenes JPG: {len(imagenes)}")

centros_x, centros_y, tiempos = [], [], []

# Elegir 5 índices aleatorios para mostrar
num_muestras = 5
indices_muestra = set(random.sample(range(len(imagenes)), min(num_muestras, len(imagenes))))

for t, img_name in enumerate(imagenes):
    img_path = os.path.join(folder, img_name)
    xs, ys_contacto, img, thresh = procesar_imagen(img_path, y_sustrato)
    if xs is None:
        print(f"No se pudo cargar la imagen {img_name}")
        continue

    #calcular centro
    centro_x, centro_y = calcular_centro(xs, ys_contacto)
    centros_x.append(centro_x)
    centros_y.append(centro_y)
    tiempos.append(t)

    #mostrar gráfico solo para los seleccionados
    if t in indices_muestra:
        graficar_tres_paneles(xs, ys_contacto, centro_x, centro_y, img, thresh, t)

# === GRAFICADO MEJORADO ===
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(tiempos, centros_y, color="#1f77b4", marker='o', linewidth=1.8, label="Centro Y (vertical)")
ax.plot(tiempos, centros_x, color="#ff7f0e", marker='s', linewidth=1.5, label="Centro X (horizontal)")
ax.set_title("Evolución del centro de la gota", fontsize=13, fontweight='bold')
ax.set_xlabel("Tiempo (frames)", fontsize=11)
ax.set_ylabel("Posición del centro [px]", fontsize=11)
ax.legend(frameon=True, loc='best')
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Gráfico de trayectoria en el plano X-Y
plt.figure(figsize=(6,6))
plt.plot(centros_x, centros_y, color="#2ca02c", marker='o', markersize=4)
plt.gca().invert_yaxis()
plt.title("Trayectoria del centro de masa", fontsize=13, fontweight='bold')
plt.xlabel("X [px]", fontsize=11)
plt.ylabel("Y [px desde el sustrato]", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()