#!/usr/bin/env python3
#
#   rrt.py
#
import matplotlib.pyplot as plt
import numpy as np
import random

from fkin3 import fkin
from fkin5 import fkin5
from segInSphere import segInSphere



from sklearn.neighbors import KDTree
from prmtools import *

from planarutils import *
import line_to_line

######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

amin, amax = -np.pi , np.pi

triangles = ((( 2, 6), ( 3, 2), ( 4, 6)),
             (( 6, 5), ( 7, 7), ( 8, 5)),
             (( 6, 9), ( 8, 9), ( 8, 7)),
             ((10, 3), (11, 6), (12, 3)))

(startx, starty) = ( 1, 5)
(goalx,  goaly)  = (13, 5)

# Number of checks with intermediate states for ConnectsTo
MAX_CHECKS = 20;

dstep = 0.25
#dstep = 0.5
#dstep = 1
dstep = 10
Nmax  = 1000

# arm link lengths
ls = np.array([0.0, 1.0, 1.0])
L  = ls.sum()

# radius of arm links
R = 0.1

# Construct the sphere (x, y, z, radius).
sphere1 = np.array([1.0, 0.0, 1.0, 0.6])
sphere2 = np.array([-1.0, 0.0, 1.0, 0.6])
sphere3 = np.array([0.0, 1.0, 1.0, 0.6])
sphere4 = np.array([0.0, -1.0, 1.0, 0.6])
obstacles = [sphere1, sphere2, sphere3, sphere4]


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
                if line_to_line.line_safe_distance(self.segments[i], self.segments[j]):
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
        numConnects = np.floor((MAX_CHECKS * d / (2 * np.pi * np.sqrt(n))))

        for alpha in range(1, numConnects + 2):
            intermediate = self.Intermediate(other, alpha / numConnects)
            if not intermediate.InFreeSpace():
                return False
        return True

######################################################################
#
#   Tree Node Definition
#
#   Define a Node class upon which to build the tree.
#
class Node:
    def __init__(self, state, parentnode):
        # Save the state matching this node.
        self.state = state

        # Link to parent for the tree structure.
        self.parent = parentnode

        # Automatically draw.
        self.Draw('r-', linewidth=1)

    # Draw a line to the parent.
    def Draw(self, *args, **kwargs):
        if self.parent is not None:
            plt.plot((self.state.x, self.parent.state.x),
                     (self.state.y, self.parent.state.y),
                     *args, **kwargs)
            plt.pause(0.001)


######################################################################
#
#   RRT Functions
#
#   Again I am distiguishing state (containing x/y information) and
#   node (containing tree structure/parent information).
#
def RRT(tree, goalstate, Nmax):
    # Loop.
    while True:
        # Determine the target state.
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        targetstate = State(x,y)        # How to pick x/y?

        # Find the nearest node (node with state nearest the target state).
        # This is inefficient (slow for large trees), but simple.
        list = [(node.state.Distance(targetstate), node) for node in tree]
        (d2, nearnode)  = min(list)
        d = np.sqrt(d2)
        nearstate = nearnode.state

        # Determine the next state, a step size (dstep) away.
        dx = targetstate.x - nearstate.x
        dy = targetstate.y - nearstate.y
        nextstate = State(nearstate.x + dstep * dx,  nearstate.y  + dstep * dy)

        # Check whether to attach (creating a new node).
        if nearstate.ConnectsTo(nextstate):
            nextnode = Node(nextstate, nearnode)
            tree.append(nextnode)

            # Also try to connect the goal.

            if  np.sqrt(nextstate.Distance(goalstate)) <= dstep:
                goalnode = Node(goalstate, nextnode)
                return(goalnode)

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return None


def targetRRT(tree, goalstate, Nmax):
    # Loop.
    while True:
        # Determine the target state.
        if random.uniform(0,1) < 0.05:
            targetstate = goalstate
        else:
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            targetstate = State(x,y)

        # Find the nearest node (node with state nearest the target state).
        # This is inefficient (slow for large trees), but simple.
        list = [(node.state.Distance(targetstate), node) for node in tree]
        (d2, nearnode)  = min(list)
        d = np.sqrt(d2)
        nearstate = nearnode.state

        # Determine the next state, a step size (dstep) away.
        dx = targetstate.x - nearstate.x
        dy = targetstate.y - nearstate.y
        nextstate = State(nearstate.x + dstep * dx,  nearstate.y  + dstep * dy)

        # Check whether to attach (creating a new node).
        if nearstate.ConnectsTo(nextstate):
            nextnode = Node(nextstate, nearnode)
            tree.append(nextnode)

            # Also try to connect the goal.

            if  np.sqrt(nextstate.Distance(goalstate)) <= dstep:
                goalnode = Node(goalstate, nextnode)
                return(goalnode)

        # Check whether we should abort (tree has gotten too large).
        if (len(tree) >= Nmax):
            return None

######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', dstep, ' and up to ', Nmax, ' nodes.')

    # Create the figure.
    Visual = Visualization()


    # Set up the start/goal states.
    startstate = State(startx, starty)
    goalstate  = State(goalx,  goaly)

    # Show the start/goal states.
    startstate.Draw('ro')
    goalstate.Draw('ro')
    Visual.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Start the tree with the start state and no parent.
    tree = [Node(startstate, None)]

    # Execute the search (return the goal leaf node).
    node = targetRRT(tree, goalstate, Nmax)

    # Check the outcome
    if node is None:
        print("UNABLE TO FIND A PATH in %d steps", Nmax)
        input("(hit return to exit)")
        return

    # Show the path.
    while node.parent is not None:
        node.Draw('b-', linewidth=2)
        node = node.parent
    print("PATH found after %d samples", len(tree))
    input("(hit return to exit)")
    return


if __name__== "__main__":
    main()
