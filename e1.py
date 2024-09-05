import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Función que define las ecuaciones diferenciales del modelo SIR
def deriv_SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parámetros del modelo
beta = 0.3  # Tasa de infección
gamma = 0.1  # Tasa de recuperación
S0 = 0.9    # Población inicial susceptible (90% de la población)
I0 = 0.1    # Población inicial infectada (10% de la población)
R0 = 0.0     # Población inicial recuperada

# Condiciones iniciales para el modelo teórico
y0 = [S0, I0, R0]

# Tiempo de simulación
t = np.linspace(0, 160, 2000)  # 160 días con 2000 pasos

# Resolver las ecuaciones diferenciales teóricas usando odeint
sol_teorico = odeint(deriv_SIR, y0, t, args=(beta, gamma))

# Extraer las soluciones teóricas
S_teorico, I_teorico, R_teorico = sol_teorico[:, 0], sol_teorico[:, 1], sol_teorico[:, 2]

# Graficar solo las curvas teóricas
plt.figure(figsize=(10,6))
plt.plot(t, S_teorico, 'b', label='S(t) - Susceptibles (Teórico)')
plt.plot(t, I_teorico, 'r', label='I(t) - Infectados (Teórico)')
plt.plot(t, R_teorico, 'g', label='R(t) - Recuperados (Teórico)')
plt.xlabel('Días')
plt.ylabel('Población (fracción del total)')
plt.title('Modelo SIR - Solución Teórica')
plt.legend()
plt.grid(True)

# Guardar la gráfica como imagen (formato PNG)
plt.savefig('modelo_SIR_solucion_teorica.png', dpi=300)

# Mostrar la gráfica en pantalla
plt.show()
