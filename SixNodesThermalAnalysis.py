#En este caso se despeja el espesor calculando el peso=area*espesor*densidad
#En esta version se utilizan los simbolos de mayor o menor de forma correcta en lugar de usar el de "Preliminary Thermal Analysis of Small Satellites"
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

k = 400 #W/(m^2 * K) conduccion entre caras
alpha = 0.96
epsilon = 0.90
sigma = const.Stefan_Boltzmann
Q_gen = 15.71 #calor generado por los componentes electronicos en W
q_solar = 1322 #W*m^2 en caso mas frio
tau = 90*60 #Periodo orbital de 90 minutos
cp = 896 
h = 400*10**3 #m
R_earth = 6873000 #m

m_total = 4 #kg

#Defino las areas
A_zen = 0.1*0.3
A_nad = 0.1*0.3
A_N = 0.1*0.3
A_S = 0.1*0.3
A_pv = 0.1*0.1 #v positivo
A_nv = 0.1*0.1 #v negativo
A_total = A_zen + A_nad + A_N + A_S + A_pv + A_nv
espesor = m_total/(A_total*2700) #La densidad del aluminio es de 2700 kg/m^3


#Relacion masa-area constante CONSULTAR SI ES ASI
m_zen = m_total * A_zen/A_total
m_nad = m_total * A_nad/A_total
m_N = m_total * A_N/A_total
m_S = m_total * A_S/A_total
m_pv = m_total * A_pv/A_total
m_nv = m_total * A_nv/A_total

############################Defino las areas de interfaz###################################
#Interfaz zenith con las otras caras
A_i_zen_pv = espesor*0.1
A_i_zen_nv = espesor*0.1
A_i_zen_N = espesor*0.3
A_i_zen_S = espesor*0.3
#Interfaz nadir con las otras caras
A_i_nad_pv = espesor*0.1
A_i_nad_nv = espesor*0.1
A_i_nad_N = espesor*0.3
A_i_nad_S = espesor*0.3
#Interfaz +v con las otras caras
A_i_pv_zen = espesor*0.1
A_i_pv_nad = espesor*0.1
A_i_pv_N = espesor*0.1
A_i_pv_S = espesor*0.1
#Interfaz -v con las otras caras
A_i_nv_zen = espesor*0.1
A_i_nv_nad = espesor*0.1
A_i_nv_N = espesor*0.1
A_i_nv_S = espesor*0.1
#Interfaz N con las otras caras
A_i_N_zen = espesor*0.3
A_i_N_nad = espesor*0.3
A_i_N_pv = espesor*0.1
A_i_N_nv = espesor*0.1
#Interfaz S con las otras caras
A_i_S_zen = espesor*0.3
A_i_S_nad = espesor*0.3
A_i_S_pv = espesor*0.1
A_i_S_nv = espesor*0.1


#######################################################################################################

#Tiempo, el periodo orbital en el paper es de 90 minutos
dt = 1
t= np.arange(0, 5*tau, dt) #5 periodos orbitales

#Rango de valoresd de Beta
beta_vals = np.linspace(0,100, 100) #Rango de valores realista era mas/menos 75

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

    
    Q_zen = F_zen*s*A_zen*q_solar*alpha + k*(A_i_zen_pv*(T_pv-T_zen) + A_i_zen_nv*(T_nv-T_zen) + A_i_zen_N*(T_N-T_zen) + A_i_zen_S*(T_S-T_zen)) - sigma*epsilon*A_zen*(T_zen**4) + Q_gen*A_zen/A_total

    Q_nad = (F_nad + a)*s*A_nad*q_solar*alpha + q_IR*A_nad + k*(A_i_nad_pv*(T_pv-T_nad) + A_i_nad_nv*(T_nv-T_nad) + A_i_nad_N*(T_N-T_nad) + A_i_nad_S*(T_S-T_nad)) - sigma*epsilon*A_nad*(T_nad**4) + Q_gen*A_nad/A_total

    Q_pv = F_pv*s*A_pv*q_solar*alpha + k*(A_i_pv_zen*(T_zen-T_pv) + A_i_pv_nad*(T_nad-T_pv) + A_i_pv_N*(T_N-T_pv) + A_i_pv_S*(T_S-T_pv)) - sigma*epsilon*A_pv*(T_pv**4) + Q_gen*A_pv/A_total

    Q_nv = F_nv*s*A_nv*q_solar*alpha + k*(A_i_nv_zen*(T_zen-T_nv) + A_i_nv_nad*(T_nad-T_nv) + A_i_nv_N*(T_N-T_nv) + A_i_nv_S*(T_S-T_nv)) - sigma*epsilon*A_nv*(T_nv**4) + Q_gen*A_nv/A_total

    Q_N = F_N*s*A_N*q_solar*alpha +  k*(A_i_N_pv*(T_pv-T_N) + A_i_N_nv*(T_nv-T_N) + A_i_N_zen*(T_zen-T_N) + A_i_N_nad*(T_nad-T_N)) - sigma*epsilon*A_N*(T_N**4) + Q_gen*A_N/A_total

    Q_S = F_S*s*A_S*q_solar*alpha + k*(A_i_S_pv*(T_pv-T_S) + A_i_S_nv*(T_nv-T_S) + A_i_S_zen*(T_zen-T_S) + A_i_S_nad*(T_nad-T_S)) - sigma*epsilon*A_S*(T_S**4) + Q_gen*A_S/A_total
    
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
surf = ax.plot_surface(B, t_T, Tzen, cmap='jet')
ax.plot_surface(B, t_T, Tzen)
ax.invert_xaxis()
ax.set_xlabel("Beta (deg)")
ax.set_ylabel("Tiempo (s)")
ax.set_zlabel("Temperatura Zenith (K)")




plt.show()





#