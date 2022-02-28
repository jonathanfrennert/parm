#!/usr/bin/env python3
#
#   prmmattress.py
#

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from sklearn.neighbors import KDTree
from planarutils import *
from prmtools import *


######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#

(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)
(amin, amax) = (0, 2*np.pi)
xw1 = 12
xw2 = 15
xw3 = 18
xw4 = 21
yw1 = 10
yw2 = 12

walls = [((xmin, ymin), (xmax, ymin)),
         ((xmax, ymin), (xmax, ymax)),
         ((xmax, ymax), (xmin, ymax)),
         ((xmin, ymax), (xmin, ymin)),
         ((xmin,  yw1), ( xw2,  yw1)),
         (( xw3,  yw1), (xmax,  yw1)),
         (( xw1,  yw2), ( xw2,  yw2)),
         (( xw2,  yw2), ( xw2, ymax))]

# Uncomment this to add in the extra wall segment.
walls.append((( xw3,  yw2), ( xw4,  yw2)))

(startx, starty, startt) = ( 5, 15, 0.5*math.pi)
(goalx,  goaly,  goalt)  = (10,  5, 0.0*math.pi)

L = 6
D = 1/2


N  = 500
K  = 25
alpha_checks = 10 # see ConnectsTo(), check (alpha - 1) intermediate states
obj_checks = 10 # see AddNodesToListObj()


######################################################################
#
#   State Definition
#
def AngleDiff(t1, t2):
    return (t1-t2) - math.pi * round((t1-t2)/math.pi)

class State:
    def __init__(self, x, y, theta):
        # Remember the (x,y,theta) position.
        self.x = x
        self.y = y
        self.t = theta

        # Pre-compute the endpoint positions.
        self.s  = math.sin(theta)
        self.c  = math.cos(theta)
        self.x1 = x + L/2 * self.c
        self.y1 = y + L/2 * self.s
        self.x2 = x - L/2 * self.c
        self.y2 = y - L/2 * self.s


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Line %5.2f,%5.2f @ %5.2f>" %
                (self.x, self.y, self.t))

    # Compute/create an intermediate state.  This can be useful if you
    # need to check the local planner by testing intermediate states.
    def Intermediate(self, other, alpha):
        return State(self.x + alpha *          (other.x - self.x),
                     self.y + alpha *          (other.y - self.y),
                     self.t + alpha * AngleDiff(other.t,  self.t))

    # Return a tuple of the coordinates.
    def Coordinates(self):
        return (self.x, self.y, L*self.s, L*self.c)


    def Segment(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]


    ############################################################
    # PRM Functions:
    # Check whether in free space.
    def InFreespace(self):
        segment = self.Segment()
        for wall in walls:
            if SegmentNearSegment(D, segment, wall):
                return False
        return True

    # Compute the relative distance to another state.
    def Distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 +
                       (self.y - other.y) ** 2 +
                       (self.s - other.s) ** 2 +
                       (self.c - other.c) ** 2)

    # Check the local planner - whether this connects to another state.
    def ConnectsTo(self, other):
        for alpha in range(1, alpha_checks):
            alpha_state = self.Intermediate(other, alpha / alpha_checks)
            if not alpha_state.InFreespace():
                return False
        return True


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self):
        # Clear and show.
        self.ClearFigure()
        self.ShowFigure()

    def ClearFigure(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_xticks([xmin, xw1, xw2, xw3, xw4, xmax])
        plt.gca().set_yticks([ymin, yw1, yw2, ymax])
        plt.gca().set_aspect('equal')

        # Show the walls.
        for wall in walls:
            plt.plot([wall[0][0], wall[1][0]],
                     [wall[0][1], wall[1][1]], 'k', linewidth=2)
        plt.plot([xw3, xw4], [yw2, yw2], 'b:', linewidth=3)

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)


    def DrawState(self, state, *args, **kwargs):
        plt.plot([state.x1, state.x2], [state.y1, state.y2], *args, **kwargs)

    def DrawLocalPath(self, head, tail, *args, **kwargs):
        n = math.ceil(head.Distance(tail) / D)
        for i in range(n+1):
            self.DrawState(head.Intermediate(tail, i/n), *args, **kwargs)


######################################################################
#
#   PRM Functions
#

#
# Sample the space uniformly
#
def AddNodesToList(nodeList, N):
    while (N > 0):
        state = State(random.uniform(xmin, xmax),
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

    # Create the figure.
    Visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(State(startx, starty, startt))
    goalnode  = Node(State(goalx,  goaly,  goalt))

    # Show the start/goal states.
    Visual.DrawState(startnode.state, 'r', linewidth=2)
    Visual.DrawState(goalnode.state,  'r', linewidth=2)
    Visual.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Create the list of sample points.
    start = time.time()
    nodeList = []
    AddNodesToListObj(nodeList, N)
    print('Sampling took ', time.time() - start)

    # Show the sample states.
    for node in nodeList:
        Visual.DrawState(node.state, 'k', linewidth=1)
    Visual.ShowFigure()
    input("Showing the nodes (hit return to continue)")

    # Add the start/goal nodes.
    nodeList.append(startnode)
    nodeList.append(goalnode)


    # Connect to the nearest neighbors.
    start = time.time()
    ConnectNearestNeighbors(nodeList, K)
    print('Connecting took ', time.time() - start)

    # Show the neighbor connections.
    #for node in nodeList:
    #    for child in node.children:
    #        Visual.DrawLocalPath(node.state, child.state, 'g-', linewidth=0.5)
    #Visual.ShowFigure()
    #input("Showing the full graph (hit return to continue)")


    # Run the A* planner.
    start = time.time()
    path = AStar(nodeList, startnode, goalnode)
    print('A* took ', time.time() - start)
    if not path:
        print("UNABLE TO FIND A PATH")
        return

    # Show the path.
    for i in range(len(path)-1):
        Visual.DrawLocalPath(path[i].state, path[i+1].state, 'r', linewidth=1)
    Visual.ShowFigure()
    input("Showing the raw path (hit return to continue)")


    # Post Process the path.
    path = PostProcess(path)

    # Show the post-processed path.
    for i in range(len(path)-1):
        Visual.DrawLocalPath(path[i].state, path[i+1].state, 'b', linewidth=2)
    Visual.ShowFigure()
    input("Showing the post-processed path (hit return to continue)")


if __name__== "__main__":
    main()
