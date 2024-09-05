import numpy as np
import matplotlib.pyplot as plt

# Función que define las ecuaciones diferenciales del modelo SIR
def deriv_SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

# Implementación del método de Runge-Kutta de orden 4 (RK4)
def runge_kutta_sir(y0, t, beta, gamma):
    n = len(t)
    y = np.zeros((n, 3))  # Matriz para almacenar S, I, R en cada paso
    y[0] = y0

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        k1 = deriv_SIR(y[i - 1], t[i - 1], beta, gamma)
        k2 = deriv_SIR(y[i - 1] + 0.5 * dt * k1, t[i - 1] + 0.5 * dt, beta, gamma)
        k3 = deriv_SIR(y[i - 1] + 0.5 * dt * k2, t[i - 1] + 0.5 * dt, beta, gamma)
        k4 = deriv_SIR(y[i - 1] + dt * k3, t[i - 1] + dt, beta, gamma)
        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return y

# Parámetros del modelo
beta = 0.1  # Tasa de infección
gamma = 0.25  # Tasa de recuperación
S0 = 0.9    # Población inicial susceptible
I0 = 2    # Población inicial infectada
R0 = 0.0     # Población inicial recuperada

# Condiciones iniciales
y0 = [S0, I0, R0]

# Tiempo de simulación
t = np.linspace(0, 100, 1000)  # 100 días con 1000 pasos

# Resolver las ecuaciones diferenciales usando Runge-Kutta
result = runge_kutta_sir(y0, t, beta, gamma)

# Extraer las soluciones S, I, R
S, I, R = result[:, 0], result[:, 1], result[:, 2]

# Graficar las curvas del modelo SIR
plt.figure(figsize=(10,6))
plt.plot(t, S, label='S(t) - Susceptibles', color='blue')
plt.plot(t, I, label='I(t) - Infectados', color='red')
plt.plot(t, R, label='R(t) - Recuperados', color='green')
plt.xlabel('Días')
plt.ylabel('Población (fracción del total)')
plt.title('Modelo SIR - Solución con Runge-Kutta 4')
plt.legend()
plt.grid(True)
plt.show()
