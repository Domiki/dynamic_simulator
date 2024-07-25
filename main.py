from vpython import *

from simulation import Simulation
from box import Box
from ball_joint import BallJoint

scene = canvas(width=800, height=800)
simul = Simulation(fps=1000)
simul.pause = True

def key_input(ev):
    s = ev.key
    if s == 'esc':
        simul.running = False
    elif s == 'p':
        simul.pause = not simul.pause

scene.bind('keydown', key_input)

box1 = Box((0, 0, 0), 1, color.green, pos_fixed=True)
box2 = Box((5, 0, 5), 1, color.blue)
box3 = Box((0, -5, 5), 1, color.red)

joint1 = BallJoint(box1, box2, (5, 0, 0), 1, color.gray(0.5))
joint2 = BallJoint(box2, box3, (5, -5, 5), 1, color.gray(0.5))

simul.add_object(box1)
simul.add_object(box2)
simul.add_object(box3)

simul.add_joint(joint1)
simul.add_joint(joint2)

while simul.running:
    if not simul.pause:
        simul.update()
    rate(simul.fps)

    