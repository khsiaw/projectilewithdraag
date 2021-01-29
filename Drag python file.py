from math import pi, radians, degrees, sin, cos, atan, sqrt, sinh, cosh, asinh
import numpy as np
from scipy.integrate import quadrature
from scipy.optimize import newton
import matplotlib.pyplot as plt
import pandas as pd

# Declaring parameters of projectile
g       = 9.81                 # Acceleration due to gravity (m/s^2)
c       = 0.47                 # Drag coefficient (spherical projectile)
r       = 0.02                 # Radius of projectile (m)
m       = 0.025                # Mass of projectile (kg)
rho_air = 1.225                # Air density (kg/m^3)
a       = pi * r**2.0          # Cross-sectional area of projectile (m^2)
m_piston = 0.10                # Mass of piston/ launching pad (kg)
k = 700                        # Spring constant in (N/m)
# Display of parameters
print('Parameters:')
print('Drag coefficient - Spherical projectile  : {:.3f}'.format(c))
print('Radius of spherical projectile (m)       : {:.3f}'.format(r))
print('Mass of projectile (kg)                  : {:.3f}'.format(m))
print('Air density (kg/m^3)                     : {:.3f}'.format(rho_air))
print('Cross-sectional area of projectile       : {:.5f}'.format(a))

#Functions
def lam(Q):
    return A - (Q + 0.5 * sinh(2.0 * Q))

def u_s(Q):
    return sqrt(g / mu) / sqrt(lam(Q))

def v_s(Q):
    return sqrt(g / mu) * sinh(Q) / sqrt(lam(Q))

def f_t(Q):
    return cosh(Q) / sqrt(lam(Q))

def f_x(Q):
    return cosh(Q) / lam(Q)

def f_y(Q):
    return sinh(2.0 * Q) / lam(Q)

def t_s(Q):
    return - quadrature(f_t, Q_0, Q, vec_func=False)[0] / sqrt(g * mu) #quadrature is for general calculations of integrations (function, start_limit, end_limit, format of display)

def x_s(Q):
    return x_0 - quadrature(f_x, Q_0, Q, vec_func=False)[0] / mu

def y_s(Q): #displacement in y direction
    return y_0 - quadrature(f_y, Q_0, Q, vec_func=False)[0] / (2.0 * mu)

def y_s_p(Q):
    return -(1.0 / (2.0 * mu)) * sinh(2.0 * Q) / lam(Q)
#Q is a quantity introduced which has no dimensions which becomes our variable

# Initial position
x_0 = 0.0
y_0 = 0.3

#for csv file
time_flight =[]
x_range =[]
max_height=[]
angles = []
ini_speed=[]
comp_range=[]

for V_0 in np.arange(4,11,1):           #loop for different speed
    for angle in np.arange(15,53,1):    #loop for different anglee in the range of 15 to 52
        psi     = radians(angle)        # Convert to radians
        u_0 = V_0 * cos(psi)
        v_0 = V_0 * sin(psi)
        
        mu = 0.5 * c * rho_air * a / m
        Q_0 = asinh(v_0 / u_0)
        A   = g / (mu * u_0**2.0) + (Q_0 + 0.5 * sinh(2.0 * Q_0))
        
        
        # Time of flight
        Q_T_est = float(asinh(-v_0 / u_0))      
        # Initial estimate for Newton's method of the value for Q for where the ball is landing
        
        Q_T = newton(y_s, Q_T_est, y_s_p)
        T = t_s(Q_T)
        
        # Horizontal range
        R = x_s(Q_T)
        
        # Maximum height
        H = y_s(0.0)
        
        #Energy needed
        Total_E = (0.5 * (m + m_piston) * V_0 **2) + ((m + m_piston)*g*0.25*sin(psi))
        compression = sqrt(2*Total_E / k)
        #print(Total_E)
        #appending the calculated values
        ini_speed.append(V_0)
        angles.append(angle)
        time_flight.append(T)
        x_range.append(R)
        max_height.append(H)
        comp_range.append(compression)
        
#Creating CSV file
# print(comp_range)
col_name = ['ini_speed','angles','x_range', 'max_height', 'time_flight']
newlist = list(zip(ini_speed, angles, x_range, max_height, time_flight))
df = pd.DataFrame(newlist, columns = col_name)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df.to_csv('Condition Table inc Drag.csv')

#
# ============================================================================================================================
# Vectorize scalar functions
t_vec = np.vectorize(t_s)
x_vec = np.vectorize(x_s)
y_vec = np.vectorize(y_s)
u_vec = np.vectorize(u_s)
v_vec = np.vectorize(v_s)

# Array for variable 'Q'
N = 101
psi_T = degrees(atan(sinh(Q_T)))
Q = np.arcsinh(np.tan(np.radians(np.linspace(degrees(psi), psi_T, N))))

# Arrays for projectile path variables
t = t_vec(Q)
x = x_vec(Q)
y = y_vec(Q)
u = u_vec(Q)
v = v_vec(Q)

# Plot of trajectory
fig, ax = plt.subplots()
line, = ax.plot(x, y, 'r-', label='Numerical')
ax.set_title(r'Projectile path')
ax.set_aspect('equal')
ax.grid(b=True)
ax.legend()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
plt.show()

# ========================================================================================================================
# Plot of velocity components
fig, ax = plt.subplots()
line, = ax.plot(t, u, 'b-', label='u')
ax.set_title(r'Horizontal velocity component')
ax.grid(b=True)
ax.legend()
ax.set_xlabel('t (s)')
ax.set_ylabel('u (m/s)')
plt.show()

fig, ax = plt.subplots()
line, = ax.plot(t, v, 'b-', label='v')
ax.set_title(r'Vertical velocity component')
ax.grid(b=True)
ax.legend()
ax.set_xlabel('t (s)')
ax.set_ylabel('v (m/s)')
plt.show()

# # ====================================================================================================================================
