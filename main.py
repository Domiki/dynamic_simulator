from vpython import *
import torch

from simulation import Simulation
from objects import *
from joints import *

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

scene = canvas(width=800, height=800)
simul = Simulation(fps=60)

def key_input(ev):
    s = ev.key
    if s == 'esc':
        simul.running = False
    elif s == 'p':
        simul.pause = not simul.pause

scene.bind('keydown', key_input)

########################## Make your own configs here ##########################
box1 = Box(simul, pos=(0, 0, 0), col=color.blue, pos_fixed=True)
box2 = Box(simul, pos=(5, 0, 0), col=color.blue)

HingeJoint(simul, box1, box2, pos=(2.5, 0, 0), axis=(0, 0, 1))
################################################################################

simul.run()