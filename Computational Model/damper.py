from IPython.terminal.debugger import display_completions_like_readline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from prompt_toolkit.widgets.dialogs import D
k_dict = {0.1E-3:1.20372919E3, 1E-3:1.41932069E3, 5E-3:1.86683541E3, 8E-3:2.10151015E3, 5.5:3.280797E3, 8:3.354818465E3} # N/m

# Defining damper class

''' Notes on this class:
      - Meant to model shock absorber object that track their own state through simulated load testing
      - Uses units of m, N 
      - Constant load rate for each test, u''' 

class damper:
  def __init__(self, d, D, mu, E, rho, n, h = 20E-3, t = 3E-3, exp = None):

    self.exp = exp # for test data

    # physical characteristics
    self.d = d # orifice diameter
    self.D = D # bellows diameter
    self.mu = mu # dynamic viscosity
    self.E = E # fluid elasticity
    self.rho = rho # fluid density
    self.n = n # number of orifices
    self.h = h # total height
    self.t = t # thickness of orifice plate

    self.beta = self.d / self.D
    self.A = np.pi/4*(D**2 - n*d**2) # plate area within bellows
    self.a = np.pi/4*(n*d**2) # orifice area

    # state vars
    self.u = 0 # load rate, m/s
    self.disp = 0 # displacement, m (assume bellows start in relaxed pos) 
    self.total_F = 0 # total force shock absorber pushes with, N
    self.energy = 0 # energy absorbed by shock absorber, N-m

    # simulation history vars
    self.disp_history = []
    self.force_history = []
    self.strain_history = []
    self.stress_history = []

    # test data
    self.test_disp = []
    self.test_force = []
    self.test_strain = []
    self.test_stress = []
    
  '''Finding damper forces'''
  # Bernoulli Assumptions: flange tappings, constant fluid speed at orifice
  def Bernoulli(self, epsilon=1): # Cd and eps should maybe be class variables
    self.Q = np.pi * (self.D/2)**2 * self.u # volumetric flowrate
    v = self.Q / (self.n*np.pi*self.d**2/4) # fluid speed at orifice
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
    qm = self.rho * self.Q/self.n # mass flowrate through orifice(s)
    self.dP = (8 * (1 - self.beta**4))/self.rho * (qm / (Cd*epsilon*np.pi*self.d**2))**2 # pressure drop across plate
    return self.dP*self.A # damping force

# Hagen-Poiseuille Assumpions: steady flow, constant cross-section
# I found number of orifices doesn't matter here, the n cancels out when pulled from Q and area terms
  def Poiseuille(self, d, L):
    self.Q = np.pi * (self.D/2)**2 * self.u # volumetric flowrate, m^3/s

    self.dP = 8 * self.mu * L * self.Q / (np.pi * (d/2)**4) # pressure drop across plate
    return self.dP*(np.pi*(d/2)**2) # damping force

  def calc_spring_F(self, disp):
    return self.k*self.disp

  '''Work with experimental data'''
  def left_point_integrate(self, x: list, y: list):
    dx = x[1:] - x[:-1]
    return np.sum(y[:-1] * dx)

  def average_trials(self, exp, speed, n_trials):
    data = []
    l_min = 1E6
    for i in range(n_trials+1)[1:]:
      path = "/content/drive/MyDrive/ME 470 Work/Data/Experiment {}/{}_{}.csv".format(exp, speed, i)
      df = pd.read_csv(path)
      test_data = df.to_numpy()[6:, :]
      test_data = test_data.astype(float)
      test_data[:,0] -= test_data[0,0] # start at disp = 0
      if test_data.shape[0] < l_min:
        l_min = test_data.shape[0]
      data.append(test_data)
    for i in range(len(data)):
      data[i] = data[i][:l_min,:]
    return sum(data)/n_trials

  def process_test(self, n_trials = 3):
    test_data = self.average_trials(self.exp, self.u, n_trials)
    test_data[:,0] -= test_data[0,0] # start at disp = 0
    test_data[:,0] *= 1E-3 # disp in m
    test_data[:,1] *= 9.81 # load in N
    self.test_disp, self.test_force = test_data[:,0], test_data[:,1] # F-D data
    self.test_strain, self.test_stress = self.test_disp/self.h, (self.test_force/self.A)*1E-3 # S-S data

  def energy_difference(self):
    exp_energy = (self.left_point_integrate(\
                     *self.process_test(self.path)) * \
                      self.A * self.h * 1E3)
    return np.abs(exp_energy - self.energy)

  def least_squares(self):
    diff = np.array(self.force_history) - \
      np.array(self.test_force)
    return np.sum(np.square(diff))

  # make sure that simulation uses disp data from test
  def match_test(self, u):
    self.u = u
    self.process_test()
    self.u *= 1e-3 # m/s
    for d in self.test_disp:
      self.step(self.u, disp = d)

  '''Plotting functions'''
  def plotFD(self, use_sim = True, use_test = True):
    if use_sim:
      plt.plot(self.disp_history, self.force_history, '-r', label = 'Simulation')
    if use_test:
      plt.scatter(self.test_disp, self.test_force, label = "Physical Data")
    plt.xlabel('Displacement (m)')
    plt.ylabel('Force (N)')

  def plotSS(self, use_sim = True, use_test = True):
    if use_sim:
      plt.plot(self.strain_history, self.stress_history, '-r', label = 'Simulation')
    if use_test:
      plt.scatter(self.test_strain, self.test_stress, label = "Physical Data")
    plt.xlabel('Strain (mm/mm)')
    plt.ylabel('Stress (kPa)')

  '''Step through simulation'''
  def step(self, u, dt = 0, disp = ''): # can define a displacement or time step
    self.u = u
    self.k = k_dict[u] + self.E*self.A/self.h # adding elasticity of toothpaste
    self.disp += u*dt
    if not disp == '': # override
      dt = (disp - self.disp)/u
      self.disp = disp
    # total force is spring + Poiseuille in bellows, in orifice, and Bernoulli
    self.total_F = self.calc_spring_F(self.disp) + self.Poiseuille(self.d, self.t)  \
              + self.Poiseuille(self.D, self.h - self.disp) + self.Bernoulli()
    self.energy += self.u*dt*self.total_F

    self.disp_history.append(self.disp)
    self.force_history.append(self.total_F)
    self.strain_history.append(self.disp/self.h)
    self.stress_history.append(self.total_F*1E-3/self.A)
