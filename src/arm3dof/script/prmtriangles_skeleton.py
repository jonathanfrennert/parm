#!/usr/bin/env python3
#
#   prmtriangles.py
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
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

triangles = ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 5), ( 7, 7), ( 8, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

(startx, starty) = ( 1, 5)
(goalx,  goaly)  = (13, 5)

N  = 350   # PICK THE PARAMERTERS
K  = 8


######################################################################
#
#   State Definition
#
class State:
    def __init__(self, x, y):
        # Remember the (x,y) position.
        self.x = x
        self.y = y


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Point %2d,%2d>" % (self.x, self.y))

    # Compute/create an intermediate state.  This can be useful if you
    # need to check the local planner by testing intermediate states.
    def Intermediate(self, other, alpha):
        return State(self.x + alpha * (other.x - self.x),
                     self.y + alpha * (other.y - self.y))

    # Return a tuple of the coordinates.
    def Coordinates(self):
        return (self.x, self.y)


    ############################################################
    # PRM Functions:
    # Check whether in free space.
    def InFreespace(self):
        for triangle in triangles:
            if PointInTriangle(self.Coordinates(), triangle):
                return False
        return True

    # Compute the relative distance to another state.
    def Distance(self, other):
        pA = self.Coordinates()
        pB = other.Coordinates()
        return np.sqrt((pA[0]-pB[0])**2 + (pA[1]-pB[1])**2)

    # Check the local planner - whether this connects to another state.
    def ConnectsTo(self, other):
        seg = [self.Coordinates(), other.Coordinates()]
        for triangle in triangles:
            if SegmentCrossTriangle(seg, triangle):
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
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for tr in triangles:
            plt.plot((tr[0][0], tr[1][0], tr[2][0], tr[0][0]),
                     (tr[0][1], tr[1][1], tr[2][1], tr[0][1]),
                     'k-', linewidth=2)

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)


    def DrawState(self, state, *args, **kwargs):
        plt.plot(state.x, state.y, *args, **kwargs)

    def DrawLocalPath(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)


######################################################################
#
#   PRM Functions
#

#
# Sample the space
#
def AddNodesToList(nodeList, N):
    # Add uniformly distributed samples
    while (N > 0):
        state = State(random.uniform(xmin, xmax),
                      random.uniform(ymin, ymax))
        if state.InFreespace():
            nodeList.append(Node(state))
            N = N-1


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
    startnode = Node(State(startx, starty))
    goalnode  = Node(State(goalx,  goaly))

    # Show the start/goal states.
    Visual.DrawState(startnode.state, 'ro')
    Visual.DrawState(goalnode.state,  'ro')
    Visual.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Create the list of sample points.
    start = time.time()
    nodeList = []
    AddNodesToList(nodeList, N)
    print('Sampling took ', time.time() - start)

    # Show the sample states.
    for node in nodeList:
        Visual.DrawState(node.state, 'kx')
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
    for node in nodeList:
        for child in node.children:
            Visual.DrawLocalPath(node.state, child.state, 'g-', linewidth=0.5)
    Visual.ShowFigure()
    input("Showing the full graph (hit return to continue)")


    # Run the A* planner.
    start = time.time()
    path = AStar(nodeList, startnode, goalnode)
    print('A* took ', time.time() - start)
    if not path:
        print("UNABLE TO FIND A PATH")
        return

    # Show the path.
    for i in range(len(path)-1):
        Visual.DrawLocalPath(path[i].state, path[i+1].state, 'r-', linewidth=1)
    Visual.ShowFigure()
    input("Showing the raw path (hit return to continue)")


    # Post Process the path.
    path = PostProcess(path)

    # Show the post-processed path.
    for i in range(len(path)-1):
        Visual.DrawLocalPath(path[i].state, path[i+1].state, 'b-', linewidth=2)
    Visual.ShowFigure()
    input("Showing the post-processed path (hit return to continue)")


if __name__== "__main__":
    main()
