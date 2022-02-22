#!/usr/bin/env python3
#
#   arm3dof.py
#
import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from sklearn.neighbors import KDTree


######################################################################
#
#   World
#   List of objects, start, goal, and parameters.
#
ls = np.array([0.0, 1.0, 1.0]) # arm link lengths
L  = ls.sum()


# Construct the walls (boxes) (x, y, z, roll, pitch, yaw, length, width, height).
floor      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*L, 2*L, 0.0])
ceiling    = np.array([0.0, 0.0, L, 0.0, 0.0, 0.0, 2*L, 2*L, 0.0])
x_pos_wall = np.array([L, 0.0, L/2, 0.0, 0.0, 0.0, 0.0, 2*L, L])
x_neg_wall = np.array([-L, 0.0, L/2, 0.0, 0.0, 0.0, 0.0, 2*L, L])
y_pos_wall = np.array([0, L, L/2, 0.0, 0.0, 0.0, 2*L, 0.0, L])
y_pos_wall = np.array([0, -L, L/2, 0.0, 0.0, 0.0, 2*L, 0.0, L])

obstacles = [floor, ceiling,
             x_pos_wall, x_neg_wall,
             y_pos_wall, y_neg_wall]


# Pick your start and goal locations (in radians).
startts = np.array([0.0, 0.0, 0.0])
goalts  = np.array([1.16, 2.36, -1.49])


# PRM parameters
N = 800
K = 50


# RRT parameters
dstep = 0.25
Nmax  = 1000


######################################################################
#
#   State Definition
#
class State:
    def __init__(self, ts):
        # Pre-compute the trigonometry.
        self.ts = ts
        self.cs = np.cos(thetas)
        self.ss = np.sin(thetas)
        self.ps = fkin(ts, cs, ss)


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        repr_str = '< - '
        for i, t in enumerate(ts):
            repr_str += f'T{i} {t * (180/np.pi):5.1f} deg - '
        return return repr_str + '>'

    # Draw where the state is:
    def Draw(self, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)
        plt.pause(0.001)


    ############################################################
    # PRM Functions:
    # Check whether in free space.
    def InFreeSpace(self):
        for wall in walls:
            if SegmentCrossBox(wall, self.box):
                return False
        return True

    # Compute the relative distance to another state.  Scale the
    # angular error by the car length.
    def Distance(self, other):
        return np.sqrt((self.x - other.x)**2 +
                       (self.y - other.y)**2 +
                       (wheelbase*AngleDiff(self.t, other.t))**2)


    ############################################################
    # RRT Functions:
    # Compute the relative distance to another state.
    def DistSquared(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)

    # Compute/create an intermediate state.
    def Intermediate(self, other, alpha):
        return State(self.x + alpha * (other.x - self.x),
                     self.y + alpha * (other.y - self.y))

    # Check the local planner - whether this connects to another state.
    def ConnectsTo(self, other):
        for triangle in triangles:
            if SegmentCrossTriangle(((self.x, self.y), (other.x, other.y)),
                                    triangle):
                return False
        return True
