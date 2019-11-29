#Animated plot of the system state on the potential energy over the simulation time

from triplepotentialMD import *
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter
writer = ImageMagickFileWriter()

#-----------------------------------------------------------------------------------
#Setup of MD_TriplePotential class instance

x_0 = 1.9
v_0 = 2.2 #2.2 #Saving initial velocities and coordinates to position and velocity vectors

MDSteps = int(500)
dt = 0.01

Test_VV3W = MD_TriplePotential(x_0, v_0, MDSteps, dt, 100)

#Running simulation and generating output arrays
for i in range(1, MDSteps):
    Test_VV3W.step(i)

x_trj = Test_VV3W.get_xarr
epot = Test_VV3W.get_epot
#-----------------------------------------------------------------------------------

# set up figure and animation

x = np.linspace(0.125, 3.5, 1000)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111, autoscale_on=False,
                     xlim=(0.125, 3.5), ylim=(-0.2, 8.0))
ax.grid()

#Build the triple well potential
A = [1.5, 1.8, 2]
a = 1.7
q = (2*math.pi)/a
potential = [(A[0]*math.cos(q*x[i]) + A[1]*math.cos(1.5*q*x[i]) + A[2]*math.sin(1.5*q*x[i]) + 4) 
      for i in np.arange(0, len(x))]

minval = [-0.2 for i in np.arange(0, len(x))]

#ax.set_title(r'Test MD', fontsize=16, weight='bold')
ax.plot(x, potential, linestyle="-", color="purple")
ax.plot(x, minval, linestyle="-", color="black")
ax.fill_between(x, minval, potential, facecolor="black")
ax.set_xlabel(r'CV Value', style='oblique', fontsize=20)
ax.set_ylabel(r'Free Energy (kJ/mol)', style='oblique', fontsize=20)
#matplotlib.rcParams['text.usetex']=False


point, = ax.plot([], [], "o", color="red", markersize=12)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""

    matplotlib.rcParams['text.usetex'] = False

    point.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return point, time_text, energy_text

def animate(i):
    """perform animation step"""
    global x_trj, epot

    new_data = (x_trj[i], epot[i]+0.15)
    print(new_data)

    point.set_data(*new_data)
    #time_text.set_text(r'X = %.1f' % new_data[0])
    #energy_text.set_text(r'Energy = %.3f kJ/mol' % epot[i])
    return point, time_text, energy_text

# choose the interval based on dt and the time to animate one step
interval = 10

ani = animation.FuncAnimation(fig, animate, frames=400,
                              interval=interval, blit=True, init_func=init)

#Save the animation to a gif
ani.save('MDhot_test.gif', writer=writer)

fig.tight_layout()

plt.show()
