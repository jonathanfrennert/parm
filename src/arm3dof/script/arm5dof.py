#!/usr/bin/env python3
#
#   arm5dof.py
#
import math
import numpy as np
import random
import time

from fkin5 import fkin5
from segInSphere import segInSphere
from line_to_line import *


from sklearn.neighbors import KDTree
from prmtools import *


######################################################################
#
#   World
#   List of objects, start, goal, and parameters.
#
amin, amax = 0 , np.pi

# old way with planes and boxes
# Construct the walls (boxes) (x, y, z, roll, pitch, yaw, length, width, height).
#floor      = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*L, 2*L, 0.0])
#ceiling    = np.array([0.0, 0.0, L, 0.0, 0.0, 0.0, 2*L, 2*L, 0.0])
#x_pos_wall = np.array([L, 0.0, L/2, 0.0, 0.0, 0.0, 0.0, 2*L, L])
#x_neg_wall = np.array([-L, 0.0, L/2, 0.0, 0.0, 0.0, 0.0, 2*L, L])
#y_pos_wall = np.array([0, L, L/2, 0.0, 0.0, 0.0, 2*L, 0.0, L])
#y_neg_wall = np.array([0, -L, L/2, 0.0, 0.0, 0.0, 2*L, 0.0, L])

#obstacles = [floor, ceiling, sphere,
#             x_pos_wall, x_neg_wall,
#             y_pos_wall, y_neg_wall]


# arm link lengths
ls = np.array([0.0, 0.5, 0.5, 0.5, 0.5])
L  = ls.sum()

# radius of arm links
R = 0.1


# Construct the sphere (x, y, z, radius).
sphere1 = np.array([1.0, 0.0, 1.0, 0.6])
sphere2 = np.array([-1.0, 0.0, 1.0, 0.6])
sphere3 = np.array([0.0, 1.0, 1.0, 0.6])
sphere4 = np.array([0.0, -1.0, 1.0, 0.6])
obstacles = [sphere1, sphere2, sphere3, sphere4]

# Pick your start and goal locations (in radians).
startts = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
goalts  = np.array([0, np.pi / 2, 0, 0, 0])


# Number of checks with intermediate states for ConnectsTo
MAX_CHECKS = 20;


# PRM default parameters
N = 800
K = 50


# RRT default parameters
dstep = 0.25
Nmax  = 1000


######################################################################
#
#   State Definition
#
class State:
    def __init__(self, ts):
        # joints
        self.ts = ts
        self.cs = np.cos(ts) # precompute
        self.ss = np.sin(ts) # precompute

        # links
        self.ps = fkin5(ls, ts, self.cs, self.ss)

        segments = []
        for i in range(len(self.ps)- 1):
            seg = np.hstack((self.ps[i],self.ps[i+1]))
            seg = np.hstack((seg, R)) # [1 x 7]
            segments.append(seg)
        self.segments = segments


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        repr_str = '< - '
        for i, t in enumerate(self.ts):
            repr_str += f'T{i} {t * (180/np.pi):5.1f} deg - '
        return repr_str + '>'


    # Return a tuple of the coordinates for KDTree.
    def Coordinates(self):
        return self.ps


    def InFreeSpace(self):
        return not (self.bodyCross() or self.obstacleCross())


    def bodyCross(self):
        for i in range(len(self.segments)):
            for j in range(i+2, len(self.segments)):
                if line_safe_distance(self.segments[i], self.segments[j]):
                    return True
        return False


    def obstacleCross(self):
        for i in range(len(self.segments)):
            for j in range(len(obstacles))  :
                if segInSphere(self.segments[i], obstacles[j]):
                    return True
        return False


    def Distance(self, other):
        return np.sqrt(np.power(np.abs(other.ts - self.ts), 2).sum())


    # Compute/create an intermediate state.
    def Intermediate(self, other, alpha):
        ts = self.ts + alpha * (other.ts - self.ts)
        return State(ts)


    # Check the local planner - whether this connects to another state.
    def ConnectsTo(self, other):
        n = len(self.ts);
        d = self.Distance(other)
        numConnects = int(np.ceil((MAX_CHECKS * d / (2 * np.pi * np.sqrt(n)))))

        for alpha in range(1, numConnects + 2):
            intermediate = self.Intermediate(other, alpha / numConnects)
            if not intermediate.InFreeSpace():
                return False
        return True

######################################################################
#
#   PRM Functions
#
#
# Sample the space uniformly
#
def AddNodesToList(nodeList, N):
    while (N > 0):
        state = State(np.array([random.uniform(amin, amax),
                      random.uniform(amin, amax),
                      random.uniform(amin, amax),
                      random.uniform(amin, amax),
                      random.uniform(amin, amax)]))
        if state.InFreeSpace():
            nodeList.append(Node(state))
            N = N-1

def AddNodesToListNearObstacle(nodeList, N):
    while (N > 0):
        state = State(np.array([random.uniform(amin, amax),
                      random.uniform(0, amax),
                      random.uniform(amin, amax),
                      random.uniform(amin, amax),
                      random.uniform(amin, amax)]))

        if state.obstacleCross():
             # Add a sample nearby until it is not in an obstacle
            while True:
                state = getNearState(state)
                
                if state.InFreeSpace():
                    nodeList.append(Node(state))
                    N = N-1 
                    break

def getNearState(state):
    maxAngleDiff = np.pi/8
    ts = np.zeros(len(state.ts))
    for i in range(len(state.ts)):

        low = state.ts[i] - maxAngleDiff
        high = state.ts[i] + maxAngleDiff 

        # The second joint has angles in [0, pi]
        if i == 1:
            if low < 0:
                low = 0
            if high > amax:
                high = amax
        # The other joints have angles between [-pi, pi]
        else:
            if low < amin:
                low = amin
            if high > amax:
                high = amax
        
        ts[i] = state.ts[i] + random.uniform(low, high)
        
    return State(ts)

#
#   Connect the nearest neighbors
#
def ConnectNearestNeighbors(nodeList, K):
    # Clear any existing neighbors.
    for node in nodeList:
        node.children = []
        node.parents  = []

    # Determine the indices for the nearest neighbors.  This also
    # reports the node itself as the closest neighbor, so add one
    # extra here and ignore the first element below.
    X   = np.array([node.state.ts for node in nodeList])
    kdt = KDTree(X)
    idx = kdt.query(X, k=(K+1), return_distance=False)

    # Add the edges (from parent to child).  Ignore the first neighbor
    # being itself.
    for i, nbrs in enumerate(idx):
        for n in nbrs[1:]:
            if nodeList[i].state.ConnectsTo(nodeList[n].state):
                nodeList[i].children.append(nodeList[n])
                nodeList[n].parents.append(nodeList[i])

#
#  Post Process the Path
#
def PostProcess(path):
    lazy_path = [path[0]]
    for i in range(1, len(path)-1):
        if not lazy_path[-1].state.ConnectsTo(path[i+1].state):
            lazy_path.append(path[i])
    lazy_path.append(path[-1])
    return lazy_path


######################################################################
#
#  Main Code
#
def plan():
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')


    # Create the start/goal nodes.
    startnode = Node(State(startts))
    goalnode  = Node(State(goalts))


    # Create the list of sample points.
    start = time.time()
    nodeList = []
    # AddNodesToListNearObstacle(nodeList, N)
    AddNodesToList(nodeList, N)
    print('Sampling took ', time.time() - start)


    # Add the start/goal nodes.
    nodeList.append(startnode)
    nodeList.append(goalnode)


    # Connect to the nearest neighbors.
    start = time.time()
    ConnectNearestNeighbors(nodeList, K)
    print('Connecting took ', time.time() - start)


    # Run the A* planner.
    start = time.time()
    path = AStar(nodeList, startnode, goalnode)
    print('A* took ', time.time() - start)
    if not path:
        print("UNABLE TO FIND A PATH")
        return
    for p in path:
        print(p.state)



    # Post Process the path.
    #path = PostProcess(path)

    return path

if __name__== "__main__":
    plan()
