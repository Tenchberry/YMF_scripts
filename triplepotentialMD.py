from typing import Optional, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import math
import scipy.stats as stats
# matplotlib.rcParams['text.usetex'] = True
import seaborn as sns

sns.set()


class MD_TriplePotential:
    """
    1D MD simulation over a triple well potential using the symplectic
    Velocity Verlet integrator

    Potential - V(x) 4 + 1.5cos(2pix/1.7) + 1.8cos(3pix/1.7) + 2sin(3pix/1.7)
    Force - F(x) = -dV(x)/dx = (6pi/17)(5sin(2pix/1.7) + 9sin(3pix/1.7) - 10cos(3pix/1.7))

    Starting coordinate fixed at 1.9, but is tunable
    """

    def __init__(self, x_i, v_i, n_steps, dt, printfreq=100, mass=1.0):
        # Initialising arrays for x, v and energies

        x_trj = np.zeros(n_steps, dtype=np.float64)
        v_trj = np.zeros(n_steps, dtype=np.float64)

        x_trj[0] = x_i
        v_trj[0] = v_i

        Epot = np.zeros(n_steps, dtype=np.float64)
        Ekin = np.zeros(n_steps, dtype=np.float64)
        Etot = np.zeros(n_steps, dtype=np.float64)

        self.x_arr = x_trj
        self.v_arr = v_trj
        self.epot = Epot
        self.ekin = Ekin
        self.etot = Etot

        self.dt = dt
        self.mass = mass
        self.freq = printfreq
        self.intervals = (0.125, 3.5)

    # Setting functions for calculation of forces, potential energy and kinetic energy

    def calc_Force(self, x, a=1.7):
        q = ((2 * math.pi) / a)

        return (6 * math.pi / 17) * (5.0 * math.sin(q * x) + 9.0 * math.sin(1.5 * q * x) - 10.0 * math.cos(1.5 * q * x))

    def calc_TriplePotential(self, x, a=1.7):
        q = ((2 * math.pi) / a)

        return 4.0 + 1.5 * math.cos(q * x) + 1.8 * math.cos(1.5 * q * x) + 2.0 * math.sin(1.5 * q * x)

    def calc_KineticEnergy(self, v):
        return 0.5 * self.mass * (v ** 2)

    # Functions for performing Velocity Verlet integrations of positions and coordinates

    def VelocityVerletUpdate_position(self, x, v, stepf=1.0):
        return x + v * self.dt * stepf

    def VelocityVerletUpdate_velocity(self, v, F, stepf=1.0):
        return v + (0.5 * self.dt * stepf) * F

    # Function for Velocity Verlet MD update

    def step(self, step):
        x = self.x_arr[step - 1]
        v = self.v_arr[step - 1]

        self.epot[step - 1] = self.calc_TriplePotential(x)
        self.ekin[step - 1] = self.calc_KineticEnergy(v)
        self.etot[step - 1] = self.ekin[step - 1] + self.epot[step - 1]

        if (step % self.freq == 0):
            print("Step %d - X = %f \n Potential = %f kJ/mol, Kinetic = %f kJ/mol, TOTAL = %f kJ/mol"
                  % (step, self.x_arr[step - 1], self.epot[step - 1], self.ekin[step - 1], self.etot[step - 1]))

        f = self.calc_Force(x)
        v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)
        x = self.VelocityVerletUpdate_position(x, v, stepf=1.0)
        f = self.calc_Force(x)
        v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)

        if (x >= self.intervals[1]) or (x<= self.intervals[0]):
            print("System is out of bounds - reverting to previous position")

            v = self.VelocityVerletUpdate_velocity(v, f, stepf=-2.5)
            x = self.VelocityVerletUpdate_position(x, v, stepf=-5.0)
            v = self.VelocityVerletUpdate_velocity(v, f, stepf=-2.5)

        self.x_arr[step] = x
        self.v_arr[step] = v

    @property
    def get_outputs(self):
        return self.x_arr, self.v_arr, self.epot, self.ekin, self.etot

    @property
    def get_xarr(self):
        return self.x_arr

    @property
    def get_varr(self):
        return self.v_arr

    @property
    def get_epot(self):
        return self.epot

    @property
    def get_ekin(self):
        return self.ekin

    @property
    def get_etot(self):
        return self.etot

class MetaD_TriplePotential(MD_TriplePotential):
        """
        1D metadynamics (MetaD) simulation over a triple well potential using the symplectic
        Velocity Verlet integrator
        
        Collective Variable (CV) for the MetaD bias is the X value for the 1D particle

        Potential - V(x) 4 + 1.5cos(2pix/1.7) + 1.8cos(3pix/1.7) + 2sin(3pix/1.7)
        Force - F(x) = -dV(x)/dx = (6pi/17)(5sin(2pix/1.7) + 9sin(3pix/1.7) - 10cos(3pix/1.7))

        Starting coordinate fixed at 1.9, but is tunable
        """
        
        def __init__(self, x_i, v_i, Gauss_width, Gauss_height,
                     n_steps, dt, printfreq=100, mass=1.0):

            super().__init__(x_i, v_i, n_steps, dt, printfreq, mass)

            #Number of points in biasgrid over which the Gaussian biases are plotted - Currently hard coded
            self.gausspoints = 100


            #MetaD Gaussian kernel parameters - Height is the maximum frequency value of the kernel
            # and width defines the kernel standard deviation
            self.gwidth = Gauss_width
            self.gheight = Gauss_height

            #Initialising grid of the explorable 1D space (CV space grid) and explorable unbiased potential
            #X_grid = np.linspace(0.125, 3.5, 100000)
            # A = [1.5, 1.8, 2]
            # a = 1.7
            # q = (2 * math.pi) / a
            # V_unbiased = [(A[0] * math.cos(q * X_grid[i]) + A[1] * math.cos(1.5 * q * X_grid[i]) + A[2] * math.sin(1.5 * q * X_grid[i]) + 4)
            #     for i in np.arange(0, len(X_grid))]
            #
            # V_grid = np.array([X_grid, V_unbiased]).T
            #
            # self.xgrid = X_grid
            # self.Vg = V_grid #Will have Gaussians added to it at each deposition step

            #Initialise dictionary of Gaussian kernels
            # - Keys - Center of the kernel
            # - Values - the density values of the kernel
            #arraylength = int(n_steps/self.freq)

            bias_dict = {}
            self.bias = bias_dict

        def __calc_biasGaussian(self, x):

            """
            Constructing the additive Gaussian bias using input width and height
            N.B. Only for listing the accumulated Gaussians into the bias grid -
            The bias potential is implemented into the dynamics using the calc_biasForce() and
            add_biasPotential() attributes

            :param x:
            :return gaussian:
            """
            gauss_range = np.linspace(x - 3 * self.gwidth, x + 3 * self.gwidth, self.gausspoints)
            gauss_ = self.gheight * stats.norm.pdf(gauss_range, x, self.gwidth)

            gaussian = np.array([gauss_range, gauss_]).T

            return gaussian

        def __add_biasPotential(self, epot):

            """
            Calculating the effective potential at t = tau_G, by adding the computed Gaussian
            kernel to the grid potential

            :param x: The center of the Gaussian kernel
            :param epot: The current potential energy
            :param gaussian: The Gaussian kernel to be added
            :return: Vg - The current effective potential
            """

            #First searching for xspace in which to deposit the Gaussian
            #for i in range(0, len(self.Vg[:,0])):
            #    if Vg

            epot_ebias = epot + self.gheight

            return epot_ebias

        def __calc_biasForce(self, x):

            """
            Gaussian bias force is calculated as the derivative of the gaussian with respect to the CV x
            Note - Implements practically the biasforce as the maximum value of the derivative -dVg/dx
            :param x:
            :return gauss_force:
            """
            gauss_range = np.linspace(x - 3 * self.gwidth, x + 3 * self.gwidth, 100)

            der_coeff = np.array([((gauss_range[i] - x)/(self.gwidth**2)) for i in range(0, len(gauss_range))]
                                 , dtype=np.float64)

            gaussforce = np.array(self.gheight * der_coeff * stats.norm.pdf(gauss_range, x, self.gwidth))
            gauss_force = np.max(gaussforce)

            return gauss_force

        def step(self, step):

            x = self.x_arr[step - 1]
            v = self.v_arr[step - 1]

            self.epot[step - 1] = self.calc_TriplePotential(x)
            self.ekin[step - 1] = self.calc_KineticEnergy(v)
            self.etot[step - 1] = self.ekin[step - 1] + self.epot[step - 1]

            if (step % self.freq == 0):
                # Bias generation - Gaussian bias potential is constructed at x(tau_G)

                self.bias[step] = self.__calc_biasGaussian(x)

                self.epot[step - 1] = self.__add_biasPotential(self.epot[step - 1])

                print("Step %d - X = %f \n Potential = %f kJ/mol, Kinetic = %f kJ/mol, TOTAL = %f kJ/mol"
                          % (step, self.x_arr[step - 1], self.epot[step - 1], self.ekin[step - 1], self.etot[step - 1]))

                f = self.calc_Force(x) + self.__calc_biasForce(x)
                v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)
                x = self.VelocityVerletUpdate_position(x, v, stepf=1.0)
                f = self.calc_Force(x) + self.__calc_biasForce(x)
                v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)

            else:

                f = self.calc_Force(x)
                v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)
                x = self.VelocityVerletUpdate_position(x, v, stepf=1.0)
                f = self.calc_Force(x)
                v = self.VelocityVerletUpdate_velocity(v, f, stepf=0.5)

            self.x_arr[step] = x
            self.v_arr[step] = v


        @property
        def get_bias(self):
            return self.bias

        # Deleting (Calling destructor)
        def __del__(self):
                print('Destructor called, Simulation terminated abnormally')















            
    