import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
k_dict = {0.1E-3:1.20372919E3, 1E-3:1.41932069E3, 5E-3:1.86683541E3, 8E-3:2.10151015E3} # N/m

# Defining damper class

''' Notes on this class:
      - Meant to model shock absorber object that track their own state through simulated load testing
      - Uses units of m, N 
      - Constant load rate for each test, u''' 

class damper:
  def __init__(self, d, D, rho, mu, n, h = 20E-3, t = 3E-3, path = ''):

    self.path = path # for test data

    # physical characteristics
    self.d = d # orifice diameter
    self.D = D # bellows diameter
    self.rho = rho # fluid density
    self.mu = mu # dynamic viscosity
    self.n = n # number of orifices
    self.h = h # total height
    self.t = t # thickness of orifice plate

    self.beta = self.d / self.D
    self.A = np.pi/4*(D**2 - n*d**2) # plate area within bellows

    # state vars
    self.u = 0 # load rate, m/s
    self.disp = 0 # displacement, m (assume bellows start in relaxed pos) 
    self.total_F = 0 # total force shock absorber pushes with, N
    self.energy = 0 # energy absorbed by shock absorber, N-m

    # history vars
    self.disp_history = []
    self.force_history = []
    self.strain_history = []
    self.stress_history = []

    
  '''Assumptions: flange tappings, constant fluid speed at orifice'''
  def Bernoulli(self, epsilon=1): # Cd and eps should maybe be class variables
    Q = np.pi * (self.D/2)**2 * self.u # volumetric flowrate
    v = Q / (self.n*np.pi*self.d**2/4) # fluid speed at orifice
    self.ReD = self.rho*v*self.D/self.mu
    
    # assume sharp edged orifice and flange tappings
    A_Cd = (19000*self.beta/self.ReD)**0.8
    L1 = 0.0254/self.D
    L2_prime = 0.0254/self.D
    M2_prime = 2*L2_prime/(1-self.beta)
    Cd = 0.5691 \
              + 0.0261*self.beta**2 \
              - 0.216*self.beta**8 \
              + 0.000521*(10**6*self.beta/self.ReD)**0.7 \
              + (0.0188 + 0.0063*A_Cd)*self.beta**(3.5)*(10**6/self.ReD)**0.3 \
              + (0.043 + 0.08*np.exp(-10*L1) - 0.123*np.exp(-7*L1))*(1 - 0.11*A_Cd)*(self.beta**4 / (1 - self.beta**4)) \
              - 0.031*(M2_prime - 0.8*M2_prime**1.1)*self.beta**1.3 \
              + 0.011*(0.75 - self.beta)*(2.8 - (self.D / 0.0254))
    qm = self.rho * Q # mass flowrate through orifice(s)
    self.dP = (8 * (1 - self.beta**4))/self.rho * (qm / (Cd*epsilon*np.pi*self.d**2))**2 # pressure drop across plate
    return self.dP*self.A # damping force

  '''using Hagen-Poiseuille, which assumes steady flow'''
  def Poiseuille(self, d, L):
    Q = np.pi * (self.D/2)**2 * self.u # volumetric flowrate, m^3/s

    self.dP = 8 * self.mu * L * Q / (np.pi * (d/2)**4) # pressure drop across plate
    return self.dP*(np.pi*(d/2)**2) # damping force

  def calc_spring_F(self, disp):
    return self.k*self.disp

  def left_point_integrate(self, x: list, y: list):
    dx = x[1:] - x[0:-1]
    return np.sum(y[:-1] * dx)

  def process_test(self, path):
    df = pd.read_csv(path)
    data = df.to_numpy()[6:, :]
    test_data = data.astype(float)
    test_data[:,0] *= 1E-3 # disp in m
    test_data[:,1] *= 9.81 # load in N
    return test_data[:,0]/self.h, (test_data[:,1]/self.A)*1E-3

  def plot_sim(self):
    plt.plot(self.strain_history, self.stress_history, label = 'Simulation')
    plt.title('Load rate = {} mm/s'.format(self.u*1E3))
    plt.xlabel('Strain (mm/mm)')
    plt.ylabel('Stress (kPa)')

  def plotSS(self):
    self.strain_history = np.array(self.disp_history)/self.h
    self.stress_history = (np.array(self.force_history)/self.A)*1E-3
    self.plot_sim()
    if self.path:
      x, y = self.process_test(self.path)
      plt.scatter(x, y, label = "Physical Data")
      plt.legend()
    plt.show()

  def step(self, dt, u):
    self.u = u
    self.k = k_dict[u]
    self.disp += u*dt

    # total force is spring + Poiseuille in bellows, in orifice, and Bernoulli
    self.total_F = self.calc_spring_F(self.disp) + self.Poiseuille(self.d, self.t)  \
              + self.Poiseuille(self.D, self.h) + self.Bernoulli()

    self.energy += self.u*dt*self.total_F
    self.disp_history.append(self.disp)
    self.force_history.append(self.total_F)