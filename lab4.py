import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros de la simulación
M, N = 50, 50  # Tamaño del grid
T = 100        # Tiempo de simulación
beta = 0.1     # Probabilidad de infección
gamma = 0.25   # Probabilidad de recuperación
rad = 1        # Radio de interacción
Nexp = 10      # Número de experimentos
I0 = 2         # Número de infectados iniciales

# Inicialización del grid con posiciones predefinidas
def inicializar_grid_con_posiciones(M, N, posiciones_infectadas):
    grid = np.zeros((M, N))
    for i, j in posiciones_infectadas:
        grid[i, j] = 1
    return grid

# Dinámica de infección y recuperación
def actualizar_grid(grid, beta, gamma, rad):
    nuevo_grid = grid.copy()
    M, N = grid.shape
    for i in range(M):
        for j in range(N):
            if grid[i, j] == 0:  # Si es susceptible
                vecinos_infectados = 0
                for di in range(-rad, rad+1):
                    for dj in range(-rad, rad+1):
                        ni, nj = (i + di) % M, (j + dj) % N
                        if grid[ni, nj] == 1:
                            vecinos_infectados += 1
                if np.random.rand() < 1 - (1 - beta)**vecinos_infectados:
                    nuevo_grid[i, j] = 1
            elif grid[i, j] == 1:  # Si es infectado
                if np.random.rand() < gamma:
                    nuevo_grid[i, j] = 2
    return nuevo_grid

# Simulación del modelo SIR con posiciones predefinidas
def simulacion_SIR_posiciones(M, N, T, posiciones_infectadas, beta, gamma, rad):
    grid = inicializar_grid_con_posiciones(M, N, posiciones_infectadas)
    historial = [grid.copy()]
    S_historial, I_historial, R_historial = [], [], []
    
    for t in range(T):
        S_historial.append(np.sum(grid == 0))
        I_historial.append(np.sum(grid == 1))
        R_historial.append(np.sum(grid == 2))
        grid = actualizar_grid(grid, beta, gamma, rad)
        historial.append(grid.copy())
    
    return historial, S_historial, I_historial, R_historial

# Repetir simulación y promediar resultados para S, I, R y grid promedio
def simulacion_promediada_con_SIR_y_grid(M, N, T, posiciones_infectadas, beta, gamma, rad, Nexp):
    S_total = np.zeros(T)
    I_total = np.zeros(T)
    R_total = np.zeros(T)
    grids_totales = np.zeros((Nexp, T + 1, M, N))  # Guardar todos los grids

    for exp in range(Nexp):
        historial, S_historial, I_historial, R_historial = simulacion_SIR_posiciones(M, N, T, posiciones_infectadas, beta, gamma, rad)
        S_total += np.array(S_historial)
        I_total += np.array(I_historial)
        R_total += np.array(R_historial)
        for t in range(T + 1):
            grids_totales[exp, t] = historial[t]
    
    # Promediar
    S_prom = S_total / Nexp
    I_prom = I_total / Nexp
    R_prom = R_total / Nexp
    grid_promedio = np.mean(grids_totales, axis=0)
    
    return S_prom, I_prom, R_prom, grid_promedio

# Graficar y guardar resultados
def graficar_y_guardar_dinamica(S, I, R, nombre_archivo):
    fig, ax = plt.subplots()
    ax.plot(S, label="Susceptibles (S)")
    ax.plot(I, label="Infectados (I)")
    ax.plot(R, label="Recuperados (R)")
    ax.set_title("Dinámica de Población Promediada (S, I, R)")
    ax.legend()
    plt.savefig(nombre_archivo)  # Guardar la gráfica
    plt.show()

# Función para animar el grid y guardar como GIF
def animar_y_guardar_grid(historial, nombre_archivo):
    fig, ax = plt.subplots()
    def actualizar_animacion(t):
        ax.clear()
        ax.imshow(historial[t], cmap='viridis', vmin=0, vmax=2)
        ax.set_title(f"Tiempo: {t}")
    
    anim = FuncAnimation(fig, actualizar_animacion, frames=len(historial), interval=200)
    
    # Guardar la animación como archivo GIF
    anim.save(nombre_archivo, writer='pillow')
    plt.show()

# Función para mostrar el grid promedio en diferentes tiempos
def mostrar_grid_promedio(grid_promedio, tiempos, nombre_archivo):
    fig, axes = plt.subplots(1, len(tiempos), figsize=(15, 5))
    for i, t in enumerate(tiempos):
        axes[i].imshow(grid_promedio[t], cmap='viridis', vmin=0, vmax=2)
        axes[i].set_title(f"Tiempo: {t}")
    plt.savefig(nombre_archivo)  # Guardar la gráfica
    plt.show()

# Menú interactivo
def mostrar_menu():
    print("Elige el ejercicio que deseas realizar:")
    print("1. Ejecutar simulación simple")
    print("2. Ejecutar simulación promediada Nexp veces")
    print("3. Experimentar con diferentes valores de beta y gamma")
    print("4. Ejecutar simulación con posiciones iniciales predefinidas y grid promedio")
    
    eleccion = int(input("Introduce el número de tu elección: "))
    
    if eleccion == 1:
        posiciones_infectadas = [(np.random.randint(0, M), np.random.randint(0, N)) for _ in range(I0)]
        historial, S_historial, I_historial, R_historial = simulacion_SIR_posiciones(M, N, T, posiciones_infectadas, beta, gamma, rad)
        graficar_y_guardar_dinamica(S_historial, I_historial, R_historial, "simulacion_simple.png")
        animar_y_guardar_grid(historial, "simulacion_simple.gif")
    
    elif eleccion == 2:
        posiciones_infectadas = [(np.random.randint(0, M), np.random.randint(0, N)) for _ in range(I0)]
        S_prom, I_prom, R_prom, _ = simulacion_promediada_con_SIR_y_grid(M, N, T, posiciones_infectadas, beta, gamma, rad, Nexp)
        graficar_y_guardar_dinamica(S_prom, I_prom, R_prom, "simulacion_promediada_dinamica.png")
    
    elif eleccion == 3:
        nuevo_beta = float(input("Introduce el valor de beta (probabilidad de infección): "))
        nuevo_gamma = float(input("Introduce el valor de gamma (probabilidad de recuperación): "))
        posiciones_infectadas = [(np.random.randint(0, M), np.random.randint(0, N)) for _ in range(I0)]
        historial, S_historial, I_historial, R_historial = simulacion_SIR_posiciones(M, N, T, posiciones_infectadas, nuevo_beta, nuevo_gamma, rad)
        graficar_y_guardar_dinamica(S_historial, I_historial, R_historial, "simulacion_experimento.png")
        animar_y_guardar_grid(historial, "simulacion_experimento.gif")

    elif eleccion == 4:
        num_infectados = int(input("Introduce el número de infectados iniciales: "))
        posiciones_infectadas = []
        print("Introduce las posiciones de los infectados (en formato i,j): ")
        for _ in range(num_infectados):
            i, j = map(int, input().split(','))
            posiciones_infectadas.append((i, j))
        
        S_prom, I_prom, R_prom, grid_promedio = simulacion_promediada_con_SIR_y_grid(M, N, T, posiciones_infectadas, beta, gamma, rad, Nexp)
        mostrar_grid_promedio(grid_promedio, [0, 50, 100], "simulacion_predefinida.png")
        graficar_y_guardar_dinamica(S_prom, I_prom, R_prom, "simulacion_predefinida_dinamica.png")
        animar_y_guardar_grid(grid_promedio, "simulacion_predefinida.gif")
        print(f"Animación guardada como simulacion_predefinida.gif y la gráfica guardada como simulacion_predefinida.png")

    else:
        print("Elección inválida, por favor elige una opción válida.")

# Ejecutar el menú interactivo repetidamente
while True:
    mostrar_menu()
    respuesta = input("¿Deseas realizar otra simulación? (s/n): ")
    if respuesta != 's':
        break
