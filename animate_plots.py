import mdtraj
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from scipy import math

import matplotlib.animation as animation
from matplotlib.animation import ImageMagickFileWriter
writer = ImageMagickFileWriter()

sns.set_style("dark")

maxr = 200

colvar = np.loadtxt("COLVAR_HMC", skiprows=31800, max_rows=maxr)
colvar_short = np.loadtxt("COLVAR_MD",skiprows=24222, max_rows=maxr)
#colvar_long = np.loadtxt("COLVAR_longTDHMC", max_rows=maxr)

fig = plt.figure(figsize=(12,6))

surface_file = np.loadtxt("fes_39.dat", usecols=(0,1,2))

phi, psi = np.meshgrid(surface_file[:,0], surface_file[:,1])
pos = np.array([surface_file[:,0], surface_file[:,1]]).T

Z = griddata(pos, surface_file[:,2], (phi, psi), method='cubic')

levels=[0, 10, 20, 30, 40, 50, 60, 70, 80]

ax = fig.add_subplot(121)
ax.set_title(r'HMC - 10fs step', fontsize=20)
HMC_plot, = ax.plot([], [], "o-", lw=0.2, color="black")
ax.contour(phi, psi, Z, levels, cmap='rainbow')
ax.contourf(phi, psi, Z, levels, cmap='rainbow')
ax.set_xlim(-math.pi, math.pi)
ax.set_ylim(-math.pi, math.pi)
ax.set_xlabel(r'$\phi$', fontsize=20)
ax.set_ylabel(r'$\psi$', fontsize=20)
ax.grid('off')

ax_md = fig.add_subplot(122)
ax_md.set_title(r'MD - Velocity Verlet (1fs)', fontsize=20)
MD_plot, = ax_md.plot([], [], "o-", lw=0.2, color="blue")
ax_md.contour(phi, psi, Z, levels, cmap='rainbow')
ax_md.contourf(phi, psi, Z, levels, cmap='rainbow')
ax_md.set_xlabel(r'$\phi$', fontsize=20)
ax_md.set_ylabel(r'$\psi$', fontsize=20)
ax_md.set_xlim(-math.pi, math.pi)
ax_md.set_ylim(-math.pi, math.pi)
ax_md.grid('off')


#ax_tmd = fig.add_subplot(133)
#ax_tmd.set_title(r'TDHMC - 700fs step', fontsize=20)
#TMD_plot, = ax_tmd.plot([], [], "o-", lw=0.2, color="blue")
#ax_tmd.contour(phi, psi, Z, levels, cmap='rainbow')
#ax_tmd.contourf(phi, psi, Z, levels, cmap='rainbow')
#ax_tmd.set_xlabel(r'$\phi$', fontsize=20)
#ax_tmd.set_ylabel(r'$\psi$', fontsize=20)
#ax_tmd.set_xlim(-math.pi, math.pi)
#ax_tmd.set_ylim(-math.pi, math.pi)
#ax_tmd.grid('off')

def init():
    """initialize animation"""

    HMC_plot.set_data([], [])
    MD_plot.set_data([], [])
    #TMD_plot.set_data([], [])
    return HMC_plot, MD_plot #, TMD_plot

def animate(i):
    """perform animation step"""
    global colvar, colvar_short # colvar_long

    HMC_data = (colvar[:i,1], colvar[:i,2])
    MD_data = (colvar_short[:i,1], colvar_short[:i,2])
    #TMD_data = (colvar_long[:i,1], colvar_long[:i,2])

    print(colvar[i,0], colvar[i,1], colvar[i,2])
    

    HMC_plot.set_data(HMC_data)
    MD_plot.set_data(MD_data)
    #TMD_plot.set_data(TMD_data)

    return HMC_plot , MD_plot #, TMD_plot

# choose the interval based on dt and the time to animate one step
interval = 10

ani = animation.FuncAnimation(fig, animate, frames=maxr,
                              interval=interval, blit=True, init_func=init)

#Save the animation to a gif
ani.save('HMCvsMD.gif', writer=writer)

fig.tight_layout()

plt.show()
