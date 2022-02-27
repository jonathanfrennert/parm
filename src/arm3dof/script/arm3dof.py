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

from fkin import fkin
from arm_link_boxes import arm_link_boxes


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
y_neg_wall = np.array([0, -L, L/2, 0.0, 0.0, 0.0, 2*L, 0.0, L])

# Construct the sphere (x, y, z, radius).
sphere = np.array([1.65, 0.0, 0.3, 0.3])

obstacles = [floor, ceiling, sphere,
             x_pos_wall, x_neg_wall,
             y_pos_wall, y_neg_wall]


# Pick your start and goal locations (in radians).
startts = np.array([0.0, 0.0, 0.0])
goalts  = np.array([1.16, 2.36, -1.49])


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
        # self.bs = arm_link_boxes() TODO


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        repr_str = '< - '
        for i, t in enumerate(self.ts):
            repr_str += f'T{i} {t * (180/np.pi):5.1f} deg - '
        return repr_str + '>'


    ############################################################
    # PRM Functions:
    # Check whether in free space.
    def InFreeSpace(self):
        return not (self.bodyCross() or self.obstacleCross())


    def bodyCross(self):
        for i in range(len(self.bs)):
            for j in range(i+1, len(self.bs)):
                if box_cross_box(self.bs[i], self.bs[j]):
                    return True
        return False


    def obstacleCross(self):
        for i in range(len(self.bs)):
            for j in range(len(self.obstacles)):
                if box_cross_box(self.bs[i], self.obstacles[j]):
                    return True
        return False


    # Compute the relative distance to another state.
    def Distance(self, other):
        return np.sqrt(np.power(AngleDiff(self.ts, other.ts), 2).sum())


def AngleDiff(t1, t2):
    return (t1-t2) - 2.0*np.pi * np.round(0.5*(t1-t2)/np.pi)


    ############################################################
    # RRT Functions:
    # Compute the relative distance to another state.
    #def DistSquared(self, other):
    #    return ((self.x - other.x)**2 + (self.y - other.y)**2)

    # Compute/create an intermediate state.
    #def Intermediate(self, other, alpha):
    #    return State(self.x + alpha * (other.x - self.x),
    #                 self.y + alpha * (other.y - self.y))

    # Check the local planner - whether this connects to another state.
    #def ConnectsTo(self, other):
    #    for triangle in triangles:
    #        if SegmentCrossTriangle(((self.x, self.y), (other.x, other.y)),
    #                                triangle):
    #            return False
    #    return True


def main():
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')


    # Create the start/goal nodes.
    #startnode = Node(State(startts))
    #goalNode  = Node(State(goalts))
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
