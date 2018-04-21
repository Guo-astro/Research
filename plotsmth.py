
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
import matplotlib.patches as patches



fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.set_ylim([-16,16])
ax1.set_xlim([-16,16])
# ax1.autoscale(enable=True, axis='y', tight=None)
f=open("/Users/guo/Research/SimulationCode/FDPS-Simulation/result/poly_force_relaxed7000.dat","r")
lines=f.readlines()
result=[]
header_line_num = 2
for line in lines[header_line_num:]:
    posx = float(line.split('\t')[3])
    posy = float(line.split('\t')[4])
    smth = 4.0*float(line.split('\t')[2])

    ax1.add_patch(
        patches.Circle(
            (posx, posy),  # (x,y)
            smth,  # radius
            fill=False
        )
    )
plt.show()
f.close()



