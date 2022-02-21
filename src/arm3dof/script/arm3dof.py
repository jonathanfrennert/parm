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
(l1, l2, l3)             = (0.0, 0.5, 0.5)              # arm link lengths
(t1, t2, t3)             = (0, 0, 0)                    # initial arm joint angles

R = l2 + l3

# Construct the walls (boxes) (x, y, z, roll, pitch, yaw, length, width, height).
walls = [ [0, 0, 0, ]
]

# Pick your start and goal locations.
(startx, starty, startt) = (2.0, 2.0, 0.0)
(goalx,  goaly,  goalt)  = (xspace + (lspace-lcar)/2 + lb, wroad + wc, 0.0)

# Spacing for drawing/testing
ddraw = 0.5

# PRM parameters
#N = 800     # 200
#K = 50      # 40
N = 200
K = 40
