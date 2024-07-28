from vpython import *
import torch

from simulation import Simulation
from objects import *
from joints import *

torch.set_printoptions(threshold=10000000, linewidth=10000000)
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

scene = canvas(width=600, height=600)
simul = Simulation(fps=75)

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
box3 = Box(simul, pos=(2, 2, 2), col=color.green)

# BallJoint(simul, box1, box2, pos=(2, 0, 0))
HingeJoint(simul, box1, box2, pos=(0, 0, 2), axis=(1, 0, 0))
FixedJoint(simul, box2, box3)
################################################################################

simul.run()