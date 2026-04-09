import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


m = 4 #kg
#beta = 
h = 400*10**3 #m
R_earth = 6378000 #m
sigma = 5.67e-8 #Cte de Stefan Boltzmann, usar scipy constants
alpha = 0.96 #absortividad
epsilon = 0.9 #emisividad
Q_gen = 15.71 #calor generado por los componentes electronicos en W
q_solar = 1322 #W*m^2 en caso mas caliente
tau = 90*60 #Periodo orbital de 90 minutos
cp = 896 # REVISAR PARA TODOS LOS ALUMINIOS

#Un satelite de 3 U tiene dimensiones de 10x10x30
A_cubo = 2*(0.10*0.10+0.10*0.30+0.10*0.30)
print(A_cubo)
radio_disco = np.sqrt(A_cubo/(4*np.pi)) #radio disco en m
#print(radio_disco) #coincide
#Segun el paper la radiacion incidente es la de un disco pero toda la superficie radiante es una superficie esferica
A_rad = A_cubo
#El area incidente va a ser la del disco, llamada A_IR
A_IR = np.pi*radio_disco**2
print(A_IR)

#Tiempo, el periodo orbital en el paper es de 90 minutos
dt = 1
t= np.arange(0, 5*tau, dt) #5 periodos orbitales

#Rango de valoresd de Beta
beta_vals = np.linspace(0,75, 75) #Rango de valores realista era mas menos 75

#Creo una matriz de temperaturas
T_s = np.zeros((len(t),len(beta_vals))) #Es una matriz de ceros de tamaño: numero de segundos x numero de betas, cada columna es un beta fijo, Consultar al profe si es lo mejor

for j, beta_deg in enumerate(beta_vals): #para que me devuelva pares j y beta
    beta = np.deg2rad(beta_deg)
    beta_cr = np.asin(R_earth/(R_earth+h)) #Ecuacion 1

    if abs(beta)<beta_cr: #Ecuacion 2
        f_E = (1/np.pi)*np.acos(np.sqrt(h**2+2*R_earth*h)/((R_earth+h)*np.cos(beta)))
    else:
        f_E = 0
    
    if beta_deg < 30: #Ecuacion 3 y 4
        a = 0.14
        q_IR = 228
    else:
        a = 0.19
        q_IR = 218

    T = np.zeros(len(t)) #Temperatura inicial para cada beta
    T[0] = 293.15 #Condicion inicial
    for i in range(len(t)-1): #es len(t) o len(t)-1? es len(t)-1 ya que estaria calculando en un instante i+1 y si llegara hasta el final entonces intentaria calcular una temperatura fuera del arreglo y da error

        if tau/2*(1 - f_E) < t[i] % tau < tau/2*(1 + f_E):
            s = 0 #eclipse
        else:
            s = 1 #iluminado

        Q_dot = q_IR*A_IR + (1+a)*q_solar*A_IR*s*alpha + Q_gen - A_rad*sigma*epsilon*T[i]**4 #Ecuacion 7
        T[i+1] = T[i] + dt/(cp*m)*Q_dot #Ecuacion 9
    T_s[:, j] = T


B, t_T = np.meshgrid(beta_vals, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(B, t_T, T_s)
ax.set_xlim(0, 75)
ax.set_ylim(0, 5*tau)
ax.invert_yaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura (K)")

plt.show()