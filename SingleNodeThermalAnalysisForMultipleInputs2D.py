import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy as sp
import json

#Datos generales
R_earth = 6378 #Km
mu = 398600 #km^3/s^2 Constante gravitatoria terrestre
sigma = const.Stefan_Boltzmann #Cte de Stefan Boltzmann


#Datos del satelite
m = 4 #kg
h = int(input("Altura del satelite en [Km]: ")) #Km
Q_gen = 15.71 #calor generado por los componentes electronicos en W
tau = (2* np.pi * ((R_earth+h)**(3/2)))/(np.sqrt(mu)) #Periodo orbital Ec. 2.64 Curtis para orbitas circulares
num_period = int(input("Cantidad de periodos orbitales: "))
############################################################ IMPORTAR MATERIAL Y TERMINACION SUPERFICIAL ######################################################
with open('Materials.json', 'r') as f:
    datos = json.load(f)

#Seleccion de material estructural del archivo json:
materials = list(datos['base_materials'].keys())
print("Materiales estructurales disponibles")
for i, mats_name in enumerate(materials):
    print(f"[{i}] {mats_name}")
sel_material = materials[int(input("Seleccione el material estructural: "))]
cp = datos['base_materials'][sel_material]['cp']

#Seleccion de la terminacion superficial del archivo json:
finishes = list(datos['surfaces_finishes'].keys())
print("Terminaciones superficiales disponibles")
for i, surf_name in enumerate(finishes):
    print(f"[{i}] {surf_name}")
sel_finish = finishes[int(input("Seleccione terminación superficial: "))]
alpha = datos['surfaces_finishes'][sel_finish]['alpha']
epsilon = datos['surfaces_finishes'][sel_finish]['epsilon']



############################################################################################################################################################


#Un satelite de 3 U tiene dimensiones de 10x10x30
A_cubo = 2*(0.10*0.10+0.10*0.30+0.10*0.30)
radio_disco = np.sqrt(A_cubo/(4*np.pi)) #radio disco en m
#Segun el paper la radiacion incidente es la de un disco pero toda la superficie radiante es una superficie esferica
A_rad = A_cubo
#El area incidente va a ser la del disco, llamada A_IR
A_IR = np.pi*radio_disco**2


#Input de angulo beta
beta_deg = float(input("Ingrese el ángulo beta en grados(0-75): "))
beta = np.deg2rad(beta_deg)

############################################################# OBTENCION DEL FLUJO SOLAR EN EL DIA #################################################################
#Segun la tabla 2.5 de Spacecraft Thermal Control Handbook: Fundamental Technologies, estos son los parametros de la Tierra en función del sol
Perihelion = 0.9833 #AU
Aphelion = 1.0167 #AU
q_solar_avg = 1367.5 #W/(AU^2*m^2) Formula 2.9 de Thermal Control Handbook
T_Earth = 365.256 #Dias
Perihelion_day = 3 #3 de enero es cuando está mas cerca del sol

#Excentricidad
e = (Aphelion-Perihelion)/(Aphelion+Perihelion) #excentricidad formula Curtis

days = np.arange(0,366)



def solar_flux(day):
    M = 2*np.pi*(day - Perihelion_day)/T_Earth #REVISAR

    E = M #PRIMERA APROXIMACION 
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M)/(1 - e * np.cos(E)) #uso un metodo iterativo porque es una ecuacion trascendental, uso el metodo de Newton Raphson del Curtis de la ec 3.17
    r = 1 - e*np.cos(E) #formula 3.25 Curtis, no uso "a" porque esta en AU y es 1
    q_solar = q_solar_avg/(r**2)
    
    return q_solar

flux = np.array([solar_flux(d) for d in days]) #flujo para todo el año

#Defino la input del usuario
day = int(input("Dia del año(1-365): "))

q_solar_day = solar_flux(day)

print("Flujo solar en el dia ", day, "es ", q_solar_day, "W/m^2")

plt.figure(1)
plt.plot(days, flux)
plt.scatter(day, q_solar_day)
plt.title("Flujo solar durante el año")
plt.xlabel("Dia del año")
plt.ylabel('Flujo solar [W/m^2]')
plt.grid(True)


############################################################# CALCULO DE TEMPERATURAS DEL SATELITE ################################################################


#Tiempo, el periodo orbital en el paper es de 90 minutos
dt = 1
t= np.arange(0, num_period*tau, dt) #5 periodos orbitales



beta_cr = np.asin((R_earth*10**3)/((R_earth*10**3)+(h*10**3))) #Ecuacion 1
q_IR = 237*((R_earth/(R_earth+h))**2)

if abs(beta)<beta_cr: #Ecuacion 2
        f_E = (1/np.pi)*np.acos(np.sqrt((h*10**3)**2+2*(R_earth*10**3)*(h*10**3))/(((R_earth*10**3)+(h*10**3))*np.cos(beta)))
else:
        f_E = 0
    
if beta_deg < 30: #Tabla 2.2 pagina 41 de Thermal Control Handbook
        a = 0.24
elif 30 < beta_deg < 60:
        a = 0.26
else:
        a = 0.26

T = np.zeros(len(t)) #Temperatura inicial para cada beta
T[0] = 293.15 #Condicion inicial

for i in range(len(t)-1): 
    if tau/2*(1 - f_E) < t[i] % tau < tau/2*(1 + f_E):
        s = 0 #eclipse
    else:
        s = 1 #iluminado

    Q_dot = q_IR*A_IR + (1+a)*q_solar_day*A_IR*s*alpha + Q_gen - A_rad*sigma*epsilon*T[i]**4 #Ecuacion 7
    T[i+1] = T[i] + dt/(cp*m)*Q_dot #Ecuacion 9


plt.figure(2)
plt.plot(t, T)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura [K]")
plt.title("Temperatura en función del tiempo")
plt.grid(True)
plt.show()