import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Настройки ---
plt.rcParams.update({'font.size': 12})

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

if os.path.exists('history/gd_history.csv'):
    path_gd = pd.read_csv('history/gd_history.csv', header=None, skiprows=1).values[:, :2]

if os.path.exists('history/ga_history.csv'):
    path_cd = pd.read_csv('history/ga_history.csv', header=None, skiprows=1).values[:, :2]

z_gd = rosenbrock(path_gd[:, 0], path_gd[:, 1])
z_cd = rosenbrock(path_cd[:, 0], path_cd[:, 1])

optimum = np.array([1.0, 1.0])
opt_z = rosenbrock(1.0, 1.0)
start_pt = path_gd[0]

fig = plt.figure(figsize=(18, 8)) # Широкое окно для двух графиков

x = np.linspace(-2.0, 2.0, 200)
y = np.linspace(-1.0, 3.0, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

ax1 = fig.add_subplot(121, projection='3d')
Z_clipped = np.clip(Z, 0, 400) # Обрезаем "горы", чтобы увидеть дно

ax1.plot_surface(X, Y, Z_clipped, cmap='jet', edgecolor='none', alpha=0.45, antialiased=True)

ax1.plot(path_gd[:, 0], path_gd[:, 1], z_gd, 'k-', linewidth=2.5, label='Градиентный спуск', zorder=10)
ax1.plot(path_cd[:, 0], path_cd[:, 1], z_cd, 'b-', linewidth=2.0, label='Ген. алгоритм', zorder=10)

ax1.scatter(optimum[0], optimum[1], opt_z, color='red', marker='*', s=300, label='Минимум (1, 1)', zorder=15)
ax1.scatter(start_pt[0], start_pt[1], z_gd[0], color='green', marker='o', s=100, label='Старт', zorder=15)

ax1.set_title("3D-поверхность с траекториями", pad=15, fontsize=14, fontweight='bold')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('f(X)')
ax1.view_init(elev=35, azim=-50) # Идеальный ракурс для этого оврага
ax1.legend(loc='upper right')

ax2 = fig.add_subplot(122)
levels = np.logspace(-1, 3, 40)
ax2.contour(X, Y, Z, levels=levels, cmap='jet', alpha=0.6, linewidths=1.5)

ax2.plot(path_gd[:, 0], path_gd[:, 1], 'k-', linewidth=2.0, label='Градиентный спуск')
ax2.plot(path_cd[:, 0], path_cd[:, 1], 'b-', linewidth=1.5, label='Ген. алгоритм', alpha=0.8)

ax2.plot(optimum[0], optimum[1], 'r*', markersize=18, label='Минимум (1, 1)')
ax2.plot(start_pt[0], start_pt[1], 'go', markersize=10, label='Старт')

ax2.set_title("Проекция на плоскость (Линии уровня)", pad=15, fontsize=14, fontweight='bold')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-1, 3])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')

# --- Вывод ---
plt.tight_layout()
plt.savefig('rosenbrock_combined.png', dpi=200, bbox_inches='tight')
plt.show()