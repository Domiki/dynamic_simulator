import torch
from vpython import *

from simulation import Simulation
from objects import *
from joints import *

torch.set_printoptions(threshold=10000000, linewidth=10000000)
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)

scene = canvas(width=800, height=800)
simul = Simulation(fps=60, pause=True)

def key_input(ev):
    s = ev.key
    if s == 'esc':
        simul.running = False
    elif s == 'p':
        simul.pause = not simul.pause
        
scene.bind('keydown', key_input)

########################## Make your own configs here ##########################
box1 = Box(simul, pos=(0, 0, 0), col=color.blue, pos_fixed=True)
box2 = Box(simul, pos=(0, 2, 2), col=color.green)

UniversalJoint(simul, box1, box2, pos=(0, 0, 2), axis=(1, 0, 0))
################################################################################

simul.run()