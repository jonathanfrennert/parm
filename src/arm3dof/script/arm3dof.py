#!/usr/bin/env python3
#
#   arm3dof.py
#
import math
import numpy as np
import random
import time

from sklearn.neighbors import KDTree
from prmtools import *


######################################################################
#
#   World
#   List of objects, start, goal, and parameters.
#
amin, amax = -np.pi , np.pi

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
ls = np.array([0.0, 1.0, 1.0])
L  = ls.sum()

# radius of arm links
R = 0.1


# Construct the sphere (x, y, z, radius).
sphere = np.array([1.65, 0.0, 0.3, 0.3])
obstacles = [sphere]


# Pick your start and goal locations (in radians).
startts = np.array([0.0, 0.0, 0.0])
goalts  = np.array([1.16, 2.36, 1.49])


# Number of checks with intermediate states for ConnectsTo
CONNECT_CHECKS = 5
ALPHAS = CONNECT_CHECKS + 2 # actuall used in ConnectsTo


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
        self.ps = fkin(ls, ts, self.cs, self.ss)
        # self.ls = arm_link_boxes() TODO


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
        return tuple(self.ps)


    def InFreeSpace(self):
        return not (self.bodyCross() or self.obstacleCross())


    def bodyCross(self):
        for i in range(len(self.ls)):
            for j in range(i+1, len(self.ls)):
                if line_to_line(self.ls[i], self.ls[j]):
                    return True
        return False


    def obstacleCross(self):
        for i in range(len(self.ls)):
            for j in range(len(obstacles)):
                if lineInSphere(self.ls[i], obstacles[j]):
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
        for alpha in range(1, ALPHAS):
            intermediate = self.Intermediate(other, alpha / connect_checks)
            if not intermediate.inFreeSpace():
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
        state = State(random.uniform(amin, amax),
                      random.uniform(ymin, ymax),
                      random.uniform(amin, amax))
        if state.InFreespace():
            nodeList.append(Node(state))
            N = N-1


#
# Sample the space with focus on edge of objects
#
def AddNodesToListObj(nodeList, N):
    while (N > 0):
        obj_state = State(random.uniform(xmin, xmax),
                          random.uniform(ymin, ymax),
                          random.uniform(amin, amax))
        if not obj_state.InFreespace():
            # if we find 1 free state close to the object state, we add it to
            # our nodes. If we do not get anything in 10 tries, we give up on
            # finding anything near that object state
            Q = 1
            num_tries = 0
            while (Q > 0) and (num_tries < obj_checks):
                r = random.gauss(2*D, 1)
                phi = random.uniform(amin, amax)
                x = obj_state.x + r * np.cos(phi)
                y = obj_state.y + r * np.sin(phi)
                theta = random.uniform(amin, amax)
                close_state = State(x, y, theta)
                if close_state.InFreespace():
                    nodeList.append(Node(close_state))
                    Q -= 1
                    N -= 1
                num_tries += 1


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
    X   = np.array([node.state.Coordinates() for node in nodeList])
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
def main():
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')


    # Create the start/goal nodes.
    startnode = Node(State(startts))
    goalNode  = Node(State(goalts))
    print(State(startts))
    print(State(goalts))
    print(State(startts).Distance(State(goalts)))


    # Show the start/goal states.
    #startnode.state.Draw(fig, 'r', linewidth=2)
    #goalnode.state.Draw(fig,  'r', linewidth=2)
    #fig.ShowFigure()
    #input("Showing basic world (hit return to continue)")


    # Create the list of sample points.
    #start = time.time()
    nodeList = []
    #AddNodesToListObj(nodeList, N)
    #print('Sampling took ', time.time() - start)

    # # Show the sample states.
    #for node in nodeList:
    #    node.state.Draw(fig, 'k', linewidth=1)
    #fig.ShowFigure()
    #input("Showing the nodes (hit return to continue)")

    # Add the start/goal nodes.
    #nodeList.append(startnode)
    #nodeList.append(goalnode)


    # Connect to the nearest neighbors.
    #start = time.time()
    #ConnectNearestNeighbors(nodeList, K)
    #print('Connecting took ', time.time() - start)

    # # Show the neighbor connections.
    # for node in nodeList:
    #     for child in node.children:
    #         plan = LocalPlan(node.state, child.state)
    #         plan.Draw(fig, 'g-', linewidth=0.5)
    # fig.ShowFigure()
    # input("Showing the full graph (hit return to continue)")


    # Run the A* planner.
    #start = time.time()
    #path = AStar(nodeList, startnode, goalnode)
    #print('A* took ', time.time() - start)
    #if not path:
    #    print("UNABLE TO FIND A PATH")
    #    return


    # Show the path.
    #DrawPath(path, fig, 'r', linewidth=1)
    #fig.ShowFigure()
    #input("Showing the raw path (hit return to continue)")

    # Post Process the path.
    #path = PostProcess(path)

    # Show the post-processed path.
    #DrawPath(path, fig, 'b', linewidth=2)
    #fig.ShowFigure()
    #input("Showing the post-processed path (hit return to continue)")


if __name__== "__main__":
    main()
