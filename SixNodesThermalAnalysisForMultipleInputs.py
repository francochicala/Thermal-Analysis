#6 Nodos para multiples inputs
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import json
import itertools

R_earth = 6378 #km
h = int(input("Altura del satelite en [Km]: ")) #Km
k = 400 #W/(m^2 * K) conduccion entre caras
mu = 398600 #km^3/s^2 Constante gravitatoria terrestre
sigma = const.Stefan_Boltzmann
Q_gen = float(input("Calor generado por los componentes electronicos en [W]: "))#15.71 #calor generado por los componentes electronicos en W
tau = (2* np.pi * ((R_earth+h)**(3/2)))/(np.sqrt(mu)) #Periodo orbital Ec. 2.64 Curtis para orbitas circulares
num_period = int(input("Cantidad de periodos orbitales: "))

############################################################ IMPORTAR MATERIAL, TERMINACION SUPERFICIAL y TAMAÑO ######################################################
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

#Selección del tamaño del cubesat
sizes = list(datos['cubesat_size'].keys())
print("Tamaños disponibles")
for i, s in enumerate(sizes):
    print(f"[{i}] {s}")
sel_size = sizes[int(input("Seleccione el tamaño del satélite: "))]
cube = datos['cubesat_size'][sel_size]

x, y, z = cube["dims"]
m_total = cube["m_total"]
espesor = cube["t_wall"]

faces = {
    "zen": ("x", "y"),
    "nad": ("x", "y"),
    "N": ("x", "z"),
    "S": ("x", "z"),
    "pv": ("y", "z"),
    "nv": ("y", "z"),
}
dims = {"x":x, "y":y, "z":z}

A = {}
for f, (d1, d2) in faces.items():
    A[f] = dims[d1]*dims[d2]
A_zen = A["zen"]
A_nad = A["nad"]
A_N   = A["N"]
A_S   = A["S"]
A_pv  = A["pv"]
A_nv  = A["nv"]

A_total = sum(A.values())

#Ahora defino las areas de interface
A_i = {}
for f1, f2 in itertools.permutations(faces.keys(),2):
    dims1 = set(faces[f1])
    dims2 = set(faces[f2])
    common = dims1.intersection(dims2) #interseccion de dims1 con dims2

    if len(common) == 1:
        edge = common.pop()
        A_i[(f1,f2)] = dims[edge] * espesor

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
    M = 2*np.pi*(day - Perihelion_day)/T_Earth 

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

#plt.figure()
#plt.plot(days, flux)
#plt.scatter(day, q_solar_day)
#plt.title("Flujo solar durante el año")
#plt.xlabel("Dia del año")
#plt.ylabel('Flujo solar [W/m^2]')
#plt.grid(True)




#Relacion masa-area constante
m_zen = m_total * A_zen/A_total
m_nad = m_total * A_nad/A_total
m_N = m_total * A_N/A_total
m_S = m_total * A_S/A_total
m_pv = m_total * A_pv/A_total
m_nv = m_total * A_nv/A_total



#######################################################################################################

#Tiempo, el periodo orbital en el paper es de 90 minutos
dt = 1
t= np.arange(0, num_period*tau, dt) #5 periodos orbitales

#Rango de valoresd de Beta
beta_vals = np.linspace(0,75, 75) #Rango de valores realista era mas/menos 75

#Como en el caso de 1 nodo, hago matrices de temperatura para cada nodo
Tzen = np.zeros((len(t), len(beta_vals)))
Tnad = np.zeros((len(t), len(beta_vals)))
TN = np.zeros((len(t), len(beta_vals)))
TS = np.zeros((len(t), len(beta_vals)))
Tpv = np.zeros((len(t), len(beta_vals)))  #v positivo
Tnv = np.zeros((len(t), len(beta_vals)))  #v negativo


#Factores de vista
def view_factors(t, beta, f_E):
    if tau/4 < t % tau < 3*tau/4:
        F_zen = 0
    else:
        F_zen = np.cos(2*np.pi*t/tau)*np.cos(beta)

    if t%tau < tau/2 * (1-f_E):
        F_nv = np.sin(2*np.pi*t/tau)*np.cos(beta)
    else:
        F_nv = 0

    if t%tau >  tau/2 * (1+f_E):
        F_pv = -np.sin(2*np.pi*t/tau)*np.cos(beta)
    else:
        F_pv = 0

    if tau/4 < t%tau < tau/2 * (1-f_E) or tau/2 * (1+f_E) < t%tau < 3*tau/4:
        F_nad = -np.cos(2*np.pi*t/tau)*np.cos(beta)
    else:
        F_nad = 0

    if tau/2*(1-f_E) < t%tau < tau/2*(1+f_E):
        F_N = 0
        F_S = 0       
    else:
        if beta>0:
            F_N = np.sin(abs(beta))
            F_S = 0
        elif beta<0:
            F_N = 0
            F_S = np.sin(abs(beta))
        else: #beta == 0
            F_N = 0
            F_S = 0        

    return F_zen, F_nad, F_pv, F_nv, F_N, F_S
 
#Modelo de transferencia de calor
def six_nodes(t, T, beta, f_E, a, q_IR):
    T_zen, T_nad, T_N, T_S, T_pv, T_nv = T #es el vector de temperaturas en un instante de tiempo, mientras que la matriz de temperatura almacenan datos, por eso son distintas las variables
    if tau/2*(1-f_E) < t%tau < tau/2*(1+f_E):
        s = 0
    else:
        s = 1 #La vuelvo a utilizar para que multiplique el albedo ya que lo necesito en el eclipse
    F_zen, F_nad, F_pv, F_nv, F_N, F_S = view_factors(t, beta, f_E)

    
    Q_zen = F_zen*s*A_zen*q_solar_day*alpha + k*(A_i[("zen","pv")]*(T_pv-T_zen) + A_i[("zen","nv")]*(T_nv-T_zen) + A_i[("zen","N")]*(T_N-T_zen) + A_i[("zen","S")]*(T_S-T_zen)) - sigma*epsilon*A_zen*(T_zen**4) + Q_gen*A_zen/A_total

    Q_nad = (F_nad + a)*s*A_nad*q_solar_day*alpha + q_IR*A_nad + k*(A_i[("nad","pv")]*(T_pv-T_nad) + A_i[("nad","nv")]*(T_nv-T_nad) + A_i[("nad","N")]*(T_N-T_nad) + A_i[("nad","S")]*(T_S-T_nad)) - sigma*epsilon*A_nad*(T_nad**4) + Q_gen*A_nad/A_total

    Q_pv = F_pv*s*A_pv*q_solar_day*alpha + k*(A_i[("pv","zen")]*(T_zen-T_pv) + A_i[("pv","nad")]*(T_nad-T_pv) + A_i[("pv","N")]*(T_N-T_pv) + A_i[("pv","S")]*(T_S-T_pv)) - sigma*epsilon*A_pv*(T_pv**4) + Q_gen*A_pv/A_total

    Q_nv = F_nv*s*A_nv*q_solar_day*alpha + k*(A_i[("nv","zen")]*(T_zen-T_nv) + A_i[("nv","nad")]*(T_nad-T_nv) + A_i[("nv","N")]*(T_N-T_nv) + A_i[("nv","S")]*(T_S-T_nv)) - sigma*epsilon*A_nv*(T_nv**4) + Q_gen*A_nv/A_total

    Q_N = F_N*s*A_N*q_solar_day*alpha +  k*(A_i[("N","pv")]*(T_pv-T_N) + A_i[("N","nv")]*(T_nv-T_N) + A_i[("N","zen")]*(T_zen-T_N) + A_i[("N","nad")]*(T_nad-T_N)) - sigma*epsilon*A_N*(T_N**4) + Q_gen*A_N/A_total

    Q_S = F_S*s*A_S*q_solar_day*alpha + k*(A_i[("S","pv")]*(T_pv-T_S) + A_i[("S","nv")]*(T_nv-T_S) + A_i[("S","zen")]*(T_zen-T_S) + A_i[("S","nad")]*(T_nad-T_S)) - sigma*epsilon*A_S*(T_S**4) + Q_gen*A_S/A_total
    
    #Ahora aplico la ecuación 8
    dT_zen = Q_zen / (m_zen*cp)
    dT_nad = Q_nad / (m_nad*cp)
    dT_N = Q_N / (m_N*cp)
    dT_S = Q_S / (m_S*cp)
    dT_pv = Q_pv / (m_pv*cp)
    dT_nv = Q_nv / (m_nv*cp)

    return np.array([dT_zen, dT_nad, dT_N, dT_S, dT_pv, dT_nv])

#Defino beta
for j, beta_deg in enumerate(beta_vals):
    beta = np.deg2rad(beta_deg)
    beta_cr = np.asin((R_earth*(10**3))/((R_earth*(10**3))+(h*(10**3)))) #Ecuacion 1
    q_IR = 237*((R_earth/(R_earth+h))**2)

    if abs(beta)<beta_cr: #Ecuacion 2
        f_E = (1/np.pi)*np.acos(np.sqrt((h*(10**3))**2+2*(R_earth*(10**3))*(h*(10**3)))/(((R_earth*(10**3))+(h*(10**3)))*np.cos(beta)))
    else:
        f_E = 0
    
    if beta_deg < 30: #Ecuacion 3 y 4
        a = 0.14
        
    else:
        a = 0.19
        

    T = np.array([293.15, 293.15, 293.15, 293.15, 293.15, 293.15])


    Tzen[0,j] = T[0]
    Tnad[0,j] = T[1]
    TN[0,j] = T[2]
    TS[0,j] = T[3]
    Tpv[0,j] = T[4]
    Tnv[0,j] = T[5]

    for i in range(len(t)-1):

        dT = six_nodes(t[i], T, beta, f_E, a, q_IR)

        T = T + dt*dT

        Tzen[i+1,j] = T[0]
        Tnad[i+1,j] = T[1]
        TN[i+1,j]   = T[2]
        TS[i+1,j]   = T[3]
        Tpv[i+1,j]  = T[4]
        Tnv[i+1,j]  = T[5]

B, t_T = np.meshgrid(beta_vals, t)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, Tzen)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura Zenith (K)")


fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, Tnad)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura Nadir (K)")

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, TN)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura Norte (K)")

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, TS)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura Sur (K)")

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, Tpv)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura +v (K)")

fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, t_T, Tnv)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura -v (K)")
plt.show()


