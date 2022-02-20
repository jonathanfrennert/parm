#!/usr/bin/env python3
#
#   rrt.py
#
import matplotlib.pyplot as plt
import numpy as np
import random

from planarutils import *


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

dstep = 0.25
Nmax  = 1000


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
    def __init__(self, x, y):
        # Remember the (x,y) position.
        self.x = x
        self.y = y


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Point %2d,%2d>" % (self.x, self.y))

    # Draw where the state is:
    def Draw(self, *args, **kwargs):
        plt.plot(self.x, self.y, *args, **kwargs)
        plt.pause(0.001)


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
        ... TODO ...
        targetstate = State(x,y)        # How to pick x/y?

        # Find the nearest node (node with state nearest the target state).
        # This is inefficient (slow for large trees), but simple.
        list = [(node.state.DistSquared(targetstate), node) for node in tree]
        (d2, nearnode)  = min(list)
        d = np.sqrt(d2)
        nearstate = nearnode.state

        # Determine the next state, a step size (dstep) away.
        ... TODO ...
        nextstate = ...

        # Check whether to attach (creating a new node).
        if nearstate.ConnectsTo(nextstate):
            nextnode = Node(nextstate, nearnode)
            tree.append(nextnode)

            # Also try to connect the goal.
            ... TODO ...
            if .... :
                ....
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
    node = RRT(tree, goalstate, Nmax)

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
