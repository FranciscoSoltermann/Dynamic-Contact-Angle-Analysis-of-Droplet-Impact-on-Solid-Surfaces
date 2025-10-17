import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import zipfile
import random  # <-- Agrega esta línea

folder = "TP4_imagenes"
frame_files = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])

# Elegir 5 índices aleatorios para mostrar
num_muestras = 5
indices_muestra = set(random.sample(range(len(frame_files)), min(num_muestras, len(frame_files))))

# ...parámetros y funciones auxiliares...
#parametros
y_sustrato = 130       # línea del sustrato
y_contact_line = 127   # línea EXACTA para tangente
spline_smooth = 0.5
degree_candidates = [3,4,5,6,7]
margin_x = 10          # margen horizontal
tangent_len = 15       # longitud de tangente (px)

#constantes
rho = 7380        # densidad [kg/m^3]
scale = 4.13e-6   # [m/px]
fps = 20538       # [frames/s]
dt = 1.0 / fps

#funciones auxiliares
def best_poly_degree(x, y, degrees):
    best_deg = degrees[0]
    min_mse = float('inf')
    for deg in degrees:
        p = np.poly1d(np.polyfit(y, x, deg))
        mse = np.mean((x - p(y))**2)
        if mse < min_mse:
            min_mse = mse
            best_deg = deg
    return best_deg

def remove_duplicate_y(y, x):
    y_unique, x_avg = [], []
    for val in np.unique(y):
        mask = (y==val)
        y_unique.append(val)
        x_avg.append(np.mean(x[mask]))
    return np.array(y_unique), np.array(x_avg)

def angle_and_tangent_at_y(spline, y_vec, y0, tangent_len):
    if y0 < y_vec.min() or y0 > y_vec.max():
        return np.nan, None, (None, None)
    x0 = float(spline(y0))
    dxdy = float(spline.derivative()(y0))
    theta = np.degrees(np.arctan2(1.0, dxdy))
    y_tan = np.linspace(y0 - tangent_len, y0 + tangent_len, 20)
    x_tan = x0 + dxdy * (y_tan - y0)
    return theta, x0, (x_tan, y_tan)

def volumen_por_revolucion(y_left, x_left, y_right, x_right):
    """ Calcula el volumen de la gota girando el perfil alrededor del eje vertical """
    y_min = max(y_left.min(), y_right.min())
    y_max = min(y_left.max(), y_right.max())
    y_common = np.linspace(y_min, y_max, 300)

    xL = np.interp(y_common, y_left, x_left)
    xR = np.interp(y_common, y_right, x_right)
    radios = (xR - xL)/2.0
    dx = np.mean(np.diff(y_common))
    vol_px3 = np.sum(np.pi * radios**2 * dx)
    return vol_px3 * (scale**3)   # [m^3]

#inicializar listas
contact_angles_left, contact_angles_right, valid_frames = [], [], []
perimetros_izq, perimetros_der, simetrias, factores_esparcimiento = [], [], [], []
energias_cineticas = []
prev_centroide = None

#procesar frames
for idx, fname in enumerate(frame_files):
    img_path = os.path.join(folder, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    #ROI hasta el sustrato
    roi = gray[:y_sustrato+1, :]
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        continue

    #contorno más grande
    contour = max(contours, key=cv2.contourArea)[:,0,:]
    x, y = contour[:,0], contour[:,1]
    mid_x = x.mean()

    #separar lados
    left, right = contour[x < mid_x], contour[x >= mid_x]
    sort_left, sort_right = np.argsort(left[:,1]), np.argsort(right[:,1])
    y_left, x_left = left[:,1][sort_left], left[:,0][sort_left]
    y_right, x_right = right[:,1][sort_right], right[:,0][sort_right]

    #eliminar duplicados
    y_left, x_left = remove_duplicate_y(y_left, x_left)
    y_right, x_right = remove_duplicate_y(y_right, x_right)

    #ajustes
    spline_left, spline_right = UnivariateSpline(y_left, x_left, s=spline_smooth), UnivariateSpline(y_right, x_right, s=spline_smooth)
    deg_left, deg_right = best_poly_degree(x_left, y_left, degree_candidates), best_poly_degree(x_right, y_right, degree_candidates)
    poly_left, poly_right = np.poly1d(np.polyfit(y_left, x_left, deg_left)), np.poly1d(np.polyfit(y_right, x_right, deg_right))

    #angulos y tangentes
    theta_left, x_contact_left, (x_tan_L, y_tan_L) = angle_and_tangent_at_y(spline_left, y_left, y_contact_line, tangent_len)
    theta_right, x_contact_right, (x_tan_R, y_tan_R) = angle_and_tangent_at_y(spline_right, y_right, y_contact_line, tangent_len)
    if not np.isnan(theta_left): theta_left = 180 - theta_left

    contact_angles_left.append(theta_left)
    contact_angles_right.append(theta_right)
    valid_frames.append(idx)

    #geometría
    perimetro_izq = np.sum(np.sqrt(np.diff(x_left)**2 + np.diff(y_left)**2))
    perimetro_der = np.sum(np.sqrt(np.diff(x_right)**2 + np.diff(y_right)**2))
    simetria = abs(perimetro_izq - perimetro_der) / ((perimetro_izq + perimetro_der)/2)
    diametro_base = x_right.max() - x_left.min()
    altura = y_contact_line - min(np.min(y_left), np.min(y_right))
    Sf = diametro_base/altura if altura > 0 else np.nan

    perimetros_izq.append(perimetro_izq)
    perimetros_der.append(perimetro_der)
    simetrias.append(simetria)
    factores_esparcimiento.append(Sf)

    #Volumen, masa, velocidad y energía
    V = volumen_por_revolucion(y_left, x_left, y_right, x_right)
    m = rho * V
    M = cv2.moments(contour)
    cx, cy = (M["m10"]/M["m00"], M["m01"]/M["m00"]) if M["m00"] != 0 else (np.mean(x), np.mean(y))
    centroide = np.array([cx, cy])
    v = 0.0
    if prev_centroide is not None:
        desplaz_px = np.linalg.norm(centroide - prev_centroide)
        desplaz_m = desplaz_px * scale
        v = desplaz_m / dt
    prev_centroide = centroide
    E_k = 0.5 * m * v**2
    energias_cineticas.append(E_k)

    #gráficos comparativos
    if idx in indices_muestra:
        y_fit_left = np.linspace(y_left.min(), max(y_left.max(), y_sustrato), 200)
        y_fit_right = np.linspace(y_right.min(), max(y_right.max(), y_sustrato), 200)
        x_fit_left_spline, x_fit_right_spline = spline_left(y_fit_left), spline_right(y_fit_right)
        x_fit_left_poly, x_fit_right_poly = poly_left(y_fit_left), poly_right(y_fit_right)

        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].plot(x_left, y_left, 'ro', label='Contorno izquierdo')
        axs[0].plot(x_right, y_right, 'bo', label='Contorno derecho')
        axs[0].plot(x_fit_left_spline, y_fit_left, 'k', linewidth=2, label='Spline')
        axs[0].plot(x_fit_right_spline, y_fit_right, 'k', linewidth=2)
        if x_contact_left is not None and x_contact_right is not None:
            axs[0].plot(x_tan_L, y_tan_L, 'g--', linewidth=2, label='Tangente')
            axs[0].plot(x_tan_R, y_tan_R, 'g--', linewidth=2)
        axs[0].set_xlim(min(x_left.min(), x_right.min())-margin_x, max(x_left.max(), x_right.max())+margin_x)
        axs[0].set_ylim(y_sustrato, 0)
        axs[0].set_aspect('equal')
        axs[0].set_title("SPLINES")
        axs[0].legend()

        axs[1].plot(x_left, y_left, 'ro', label='Contorno izquierdo')
        axs[1].plot(x_right, y_right, 'bo', label='Contorno derecho')
        axs[1].plot(x_fit_left_poly, y_fit_left, 'k--', linewidth=2, label='Polinomio')
        axs[1].plot(x_fit_right_poly, y_fit_right, 'k--', linewidth=2)
        if x_contact_left is not None and x_contact_right is not None:
            axs[1].plot(x_tan_L, y_tan_L, 'g--', linewidth=2, label='Tangente')
            axs[1].plot(x_tan_R, y_tan_R, 'g--', linewidth=2)
        axs[1].set_xlim(min(x_left.min(), x_right.min())-margin_x, max(x_left.max(), x_right.max())+margin_x)
        axs[1].set_ylim(y_sustrato, 0)
        axs[1].set_aspect('equal')
        axs[1].set_title("POLINOMIOS")
        axs[1].legend()

        plt.suptitle(f"Frame {idx+1}: Ángulo izq={theta_left:.1f}°, der={theta_right:.1f}°\n"
                     f"Perímetros: izq={perimetro_izq:.1f}, der={perimetro_der:.1f}, "
                     f"Simetría={simetria:.3f}, Sf={Sf:.3f}, E_k={E_k:.3e} J", fontsize=11)
        plt.tight_layout()
        plt.show()

    print(f"Frame {idx+1}: Ángulo izq={theta_left:.1f}°, der={theta_right:.1f}°, "
          f"P_izq={perimetro_izq:.1f}, P_der={perimetro_der:.1f}, Sim={simetria:.3f}, "
          f"Sf={Sf:.3f}, E_k={E_k:.3e} J")

# === GRAFICADO ESTILO OSCURO ===
plt.style.use('dark_background')

# Ángulos de contacto
plt.figure(figsize=(9,5))
plt.plot(valid_frames, contact_angles_left, 'o-', color='#00FFFF', label='Izquierdo', linewidth=2)
plt.plot(valid_frames, contact_angles_right, 's-', color='#FF6EC7', label='Derecho', linewidth=2)
plt.title("Evolución del ángulo de contacto", fontsize=14, fontweight='bold')
plt.xlabel("Frame", fontsize=11)
plt.ylabel("Ángulo [°]", fontsize=11)
plt.grid(alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Simetría
plt.figure(figsize=(9,5))
plt.plot(valid_frames, simetrias, '^-', color='#FFD700', linewidth=2)
plt.title("Índice de simetría en el tiempo", fontsize=14, fontweight='bold')
plt.xlabel("Frame", fontsize=11)
plt.ylabel("Simetría relativa", fontsize=11)
plt.grid(alpha=0.4, linestyle=':')
plt.tight_layout()
plt.show()

# Factor de esparcimiento
plt.figure(figsize=(9,5))
plt.plot(valid_frames, factores_esparcimiento, 'd-', color='#00FF7F', linewidth=2)
plt.title("Factor de esparcimiento (Sf = D_base / H)", fontsize=14, fontweight='bold')
plt.xlabel("Frame", fontsize=11)
plt.ylabel("Sf", fontsize=11)
plt.grid(alpha=0.4, linestyle=':')
plt.tight_layout()
plt.show()

# Energía cinética (si existe)
if 'energias_cineticas' in locals() and len(energias_cineticas) == len(valid_frames):
    plt.figure(figsize=(9,5))
    plt.plot(valid_frames, energias_cineticas, 'o-', color='#FF6347', linewidth=2)
    plt.title("Energía cinética de la gota", fontsize=14, fontweight='bold')
    plt.xlabel("Frame", fontsize=11)
    plt.ylabel("E_k [J]", fontsize=11)
    plt.grid(alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.show()

