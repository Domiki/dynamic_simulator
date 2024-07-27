from vpython import *
import torch

from simulation import Simulation
from objects import *
from joints import *

torch.set_printoptions(threshold=10000000, linewidth=10000000)
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float64)

scene = canvas(width=600, height=600)
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
box2 = Box(simul, pos=(2, 0, 0), col=color.green)
box3 = Box(simul, pos=(-2, 0, 0), col=color.green)
box4 = Box(simul, pos=(0, 2, 0), col=color.green)
box5 = Box(simul, pos=(0, -2, 0), col=color.green)
box6 = Box(simul, pos=(0, 0, 2), col=color.green)
box7 = Box(simul, pos=(0, 0, -2), col=color.green)

BallJoint(simul, box1, box2, pos=(1, 0, 0))
BallJoint(simul, box1, box3, pos=(-1, 0, 0))
BallJoint(simul, box1, box4, pos=(0, 1, 0))
BallJoint(simul, box1, box5, pos=(0, -1, 0))
BallJoint(simul, box1, box6, pos=(0, 0, 1))
BallJoint(simul, box1, box7, pos=(0, 0, -1))
################################################################################

simul.run()