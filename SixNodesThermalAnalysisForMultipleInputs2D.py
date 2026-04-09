#6 Nodos para multiples inputs
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import json
import itertools

R_earth = 6378 #km
h = int(input("Altura del satelite en [Km]: ")) #Km
k = int(input("Valor de conducción entre caras en [W/(m^2*K)]: ")) #400 #W/(m^2 * K) conduccion entre caras
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
print("Terminaciones superficiales disponibles EXTERIOR")
for i, surf_name in enumerate(finishes):
    print(f"[{i}] {surf_name}")
sel_finish = finishes[int(input("Seleccione terminación superficial EXTERIOR: "))]
alpha = datos['surfaces_finishes'][sel_finish]['alpha']
epsilon = datos['surfaces_finishes'][sel_finish]['epsilon']

print("\nTerminaciones superficiales disponibles INTERIOR:")
for i, name in enumerate(finishes):
    print(f"  [{i}] {name}")
sel_finish_int = finishes[int(input("Seleccione terminación superficial INTERIOR: "))]
epsilon_int = datos['surfaces_finishes'][sel_finish_int]['epsilon']


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

############################################################# OBTENCIÓN DE FACTORES DE VISTA INTERNOS ENTRE CARAS ###############
def F_parallel(a, b, c):
    X = a/c
    Y = b/c
    F = (2/(np.pi * X * Y)) * (np.log(np.sqrt((1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2) )) + X * np.sqrt(1 + Y**2) * np.arctan(X / np.sqrt(1+Y**2)) + (Y * np.sqrt(1 + X**2)) * np.arctan(Y / np.sqrt(1 + X**2)) - X * np.arctan(X) - Y * np.arctan(Y) ) #Appendix A Common View Factor Tables de Introduction to Spacecraft Thermal Design
    
    return F

def F_perp(a, b, c):
    H = b / a
    W = c / a
    F = (1 / (np.pi * W)) * (W * np.arctan(1 / W) + H * np.arctan(1 / H) - np.sqrt(H**2 + W**2) * np.arctan(1 / np.sqrt(H**2 + W**2)) + 0.25 * np.log(((1 + W**2) * (1 + H**2) / (1 + W**2 + H**2)) * ((W**2 * (1 + W**2 + H**2)) / ((1 + W**2) * (W**2 + H**2))) ** (W**2) * ((H**2 * (1 + W**2 + H**2)) / ((1 + H**2) * (W**2 + H**2))) ** (H**2)))
    return F

# Caras paralelas opuestas
F_zen_nad = F_parallel(x, y, z)   # separadas z
F_pv_nv   = F_parallel(x, z, y)   # separadas y
F_N_S     = F_parallel(y, z, x)   # separadas x

# Caras perpendiculares con borde común
# Notación: F_perp(borde_comun, prof_cara1, prof_cara2)
F_zen_N  = F_perp(x, y, z)   # zen-N:  borde x, zen profund y, N profund z
F_zen_S  = F_perp(x, y, z)   
F_zen_pv = F_perp(y, x, z)   # zen-pv: borde y, zen profund x, pv profund z
F_zen_nv = F_perp(y, x, z)   
F_nad_N  = F_perp(x, y, z)
F_nad_S  = F_perp(x, y, z)
F_nad_pv = F_perp(y, x, z)
F_nad_nv = F_perp(y, x, z)
F_N_pv   = F_perp(z, x, y)   # N-pv:  borde z, N profund x, pv profund y
F_N_nv   = F_perp(z, x, y)
F_S_pv   = F_perp(z, x, y)
F_S_nv   = F_perp(z, x, y)

#Factores de vista internos F[i->j]
F_int = {
    ("zen", "nad"): F_zen_nad,  ("nad", "zen"): F_zen_nad,
    ("pv",  "nv"):  F_pv_nv,   ("nv",  "pv"):  F_pv_nv,
    ("N",   "S"):   F_N_S,     ("S",   "N"):   F_N_S,
    ("zen", "N"):   F_zen_N,   ("N",   "zen"): F_zen_N,
    ("zen", "S"):   F_zen_S,   ("S",   "zen"): F_zen_S,
    ("zen", "pv"):  F_zen_pv,  ("pv",  "zen"): F_zen_pv,
    ("zen", "nv"):  F_zen_nv,  ("nv",  "zen"): F_zen_nv,
    ("nad", "N"):   F_nad_N,   ("N",   "nad"): F_nad_N,
    ("nad", "S"):   F_nad_S,   ("S",   "nad"): F_nad_S,
    ("nad", "pv"):  F_nad_pv,  ("pv",  "nad"): F_nad_pv,
    ("nad", "nv"):  F_nad_nv,  ("nv",  "nad"): F_nad_nv,
    ("N",   "pv"):  F_N_pv,    ("pv",  "N"):   F_N_pv,
    ("N",   "nv"):  F_N_nv,    ("nv",  "N"):   F_N_nv,
    ("S",   "pv"):  F_S_pv,    ("pv",  "S"):   F_S_pv,
    ("S",   "nv"):  F_S_nv,    ("nv",  "S"):   F_S_nv,
}


epsilon_int_dict = {face: epsilon_int for face in faces}

############################################################# TRANSFERENCIA RADIATIVA INTERNA ######################################
def epsilon_eff(eps_i, eps_j): #Obtenido de la diapositiva 26 de la clase
    return (eps_i * eps_j) / (eps_i + eps_j - eps_i * eps_j)
 
 
def Q_rad_int(face, T_dict): #T_dict es un diccionario de temperaturas que convierte las variables individuales a indexadas por nombre de cara para poder iterar sobre ellas con Q_rad_int
    Q   = 0.0
    A_f = A[face]
    T_f = T_dict[face]
    eps_i = epsilon_int_dict[face]
 
    for other in faces:
        if other == face:
            continue
        key = (face, other)
        if key in F_int:
            eps_j  = epsilon_int_dict[other]
            eps_ij = epsilon_eff(eps_i, eps_j)
            Q += sigma * A_f * eps_ij * F_int[key] * (T_dict[other]**4 - T_f**4)
    return Q

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



#Relacion masa-area constante
m_zen = m_total * A_zen/A_total
m_nad = m_total * A_nad/A_total
m_N = m_total * A_N/A_total
m_S = m_total * A_S/A_total
m_pv = m_total * A_pv/A_total
m_nv = m_total * A_nv/A_total



#######################################################################################################


dt = 1
t= np.arange(0, num_period*tau, dt) 


#Input de angulo beta
beta_deg = float(input("Ingrese el ángulo beta en grados(0-90): "))
beta = np.deg2rad(beta_deg)

#Como en el caso de 1 nodo, hago matrices de temperatura para cada nodo
Tzen = np.zeros(len(t))
Tnad = np.zeros(len(t))
TN = np.zeros(len(t))
TS = np.zeros(len(t))
Tpv = np.zeros(len(t))  #v positivo
Tnv = np.zeros(len(t))  #v negativo


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
    T_dict = {"zen": T_zen, "nad": T_nad, "N": T_N, "S": T_S, "pv": T_pv, "nv": T_nv}
    if tau/2*(1-f_E) < t%tau < tau/2*(1+f_E):
        s = 0
    else:
        s = 1 #La vuelvo a utilizar para que multiplique el albedo ya que lo necesito en el eclipse
    F_zen, F_nad, F_pv, F_nv, F_N, F_S = view_factors(t, beta, f_E)

    
    Q_zen = F_zen*s*A_zen*q_solar_day*alpha + k*(A_i[("zen","pv")]*(T_pv-T_zen) + A_i[("zen","nv")]*(T_nv-T_zen) + A_i[("zen","N")]*(T_N-T_zen) + A_i[("zen","S")]*(T_S-T_zen)) - sigma*epsilon*A_zen*(T_zen**4) + Q_gen*A_zen/A_total + Q_rad_int("zen", T_dict)

    Q_nad = (F_nad + a)*s*A_nad*q_solar_day*alpha + q_IR*A_nad + k*(A_i[("nad","pv")]*(T_pv-T_nad) + A_i[("nad","nv")]*(T_nv-T_nad) + A_i[("nad","N")]*(T_N-T_nad) + A_i[("nad","S")]*(T_S-T_nad)) - sigma*epsilon*A_nad*(T_nad**4) + Q_gen*A_nad/A_total + Q_rad_int("nad", T_dict)

    Q_pv = F_pv*s*A_pv*q_solar_day*alpha + k*(A_i[("pv","zen")]*(T_zen-T_pv) + A_i[("pv","nad")]*(T_nad-T_pv) + A_i[("pv","N")]*(T_N-T_pv) + A_i[("pv","S")]*(T_S-T_pv)) - sigma*epsilon*A_pv*(T_pv**4) + Q_gen*A_pv/A_total + Q_rad_int("pv", T_dict)

    Q_nv = F_nv*s*A_nv*q_solar_day*alpha + k*(A_i[("nv","zen")]*(T_zen-T_nv) + A_i[("nv","nad")]*(T_nad-T_nv) + A_i[("nv","N")]*(T_N-T_nv) + A_i[("nv","S")]*(T_S-T_nv)) - sigma*epsilon*A_nv*(T_nv**4) + Q_gen*A_nv/A_total + Q_rad_int("nv", T_dict)

    Q_N = F_N*s*A_N*q_solar_day*alpha +  k*(A_i[("N","pv")]*(T_pv-T_N) + A_i[("N","nv")]*(T_nv-T_N) + A_i[("N","zen")]*(T_zen-T_N) + A_i[("N","nad")]*(T_nad-T_N)) - sigma*epsilon*A_N*(T_N**4) + Q_gen*A_N/A_total + Q_rad_int("N", T_dict)

    Q_S = F_S*s*A_S*q_solar_day*alpha + k*(A_i[("S","pv")]*(T_pv-T_S) + A_i[("S","nv")]*(T_nv-T_S) + A_i[("S","zen")]*(T_zen-T_S) + A_i[("S","nad")]*(T_nad-T_S)) - sigma*epsilon*A_S*(T_S**4) + Q_gen*A_S/A_total + Q_rad_int("S", T_dict)
    
    #Ahora aplico la ecuación 8
    dT_zen = Q_zen / (m_zen*cp)
    dT_nad = Q_nad / (m_nad*cp)
    dT_N = Q_N / (m_N*cp)
    dT_S = Q_S / (m_S*cp)
    dT_pv = Q_pv / (m_pv*cp)
    dT_nv = Q_nv / (m_nv*cp)

    return np.array([dT_zen, dT_nad, dT_N, dT_S, dT_pv, dT_nv])

#Defino beta

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


Tzen[0] = T[0]
Tnad[0] = T[1]
TN[0] = T[2]
TS[0] = T[3]
Tpv[0] = T[4]
Tnv[0] = T[5]

for i in range(len(t)-1):

    dT = six_nodes(t[i], T, beta, f_E, a, q_IR)

    T = T + dt*dT

    Tzen[i+1] = T[0]
    Tnad[i+1] = T[1]
    TN[i+1]   = T[2]
    TS[i+1]   = T[3]
    Tpv[i+1]  = T[4]
    Tnv[i+1]  = T[5]


plt.figure(1)
plt.plot(t, Tzen)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura Zenith [K]")
plt.title("Temperatura Zenith vs tiempo")
plt.grid(True)

plt.figure(2)
plt.plot(t, Tnad)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura Nadir [K]")
plt.title("Temperatura Nadir vs tiempo")
plt.grid(True)

plt.figure(3)
plt.plot(t, TN)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura Norte [K]")
plt.title("Temperatura Norte vs tiempo")
plt.grid(True)

plt.figure(4)
plt.plot(t, TS)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura Sur [K]")
plt.title("Temperatura Sur vs tiempo")
plt.grid(True)

plt.figure(5)
plt.plot(t, Tpv)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura +V [K]")
plt.title("Temperatura +V vs tiempo")
plt.grid(True)

plt.figure(6)
plt.plot(t, Tnv)
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura -V [K]")
plt.title("Temperatura -V vs tiempo")
plt.grid(True)
plt.show()


