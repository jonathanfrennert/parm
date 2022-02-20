#!/usr/bin/env python3
#
#   testplanarutils.py
#
#   Test the planar utilities
#
import numpy as np
import matplotlib.pyplot as plt

from planarutils import *


######################################################################
#
#   General/World Definitions
#

# Define the size.
(xmin, xmax) = (-3, 3)
(ymin, ymax) = (-2, 2)

# Define the central segment and triangle.
segment  = ((-1, 0), (1, 0))
triangle = ((-1, 0), (1, 0), (0, 1))
box      = ((-1, 0), (1, 0), (1, 1), (-1, 1))
arc      = ((0,0), (1, 0), (0, 1))

# Define the test points.
D = 0.1
xlist = np.arange(xmin, xmax, D)
ylist = np.arange(ymin, ymax, D)
d = D/2 + 0.001


pointlist = [(x,y)
             for x in np.arange(xmin, xmax, D)
             for y in np.arange(ymin, ymax, D)]

deltalist = [(np.cos(alpha)+1e-9, np.sin(alpha))
             for alpha in np.arange(1e-10, 2*np.pi, np.pi/10-1e-8)]


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

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)


    def DrawPoint(self, p):
        plt.plot(p[0], p[1], 'ro')

    def DrawSegment(self, s):
        plt.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], 'k-', linewidth=2)

    def DrawTriangle(self, t):
        self.DrawSegment((t[0], t[1]))
        self.DrawSegment((t[1], t[2]))
        self.DrawSegment((t[2], t[0]))

    def DrawBox(self, b):
        self.DrawSegment((b[0], b[1]))
        self.DrawSegment((b[1], b[2]))
        self.DrawSegment((b[2], b[3]))
        self.DrawSegment((b[3], b[0]))

    def DrawArc(self, a):
        (xc, yc) = (a[0][0]     , a[0][1])
        (xa, ya) = (a[1][0] - xc, a[1][1] - yc)
        (xb, yb) = (a[2][0] - xc, a[2][1] - yc)
        alpha = np.arctan2(xa*yb-ya*xb, xa*xb+ya*yb)
        x = [(xc + np.cos(dalpha)*xa - np.sin(dalpha)*ya)
             for dalpha in np.arange(0.0, alpha, 0.01*alpha)]
        y = [(xc + np.cos(dalpha)*ya + np.sin(dalpha)*xa)
             for dalpha in np.arange(0.0, alpha, 0.01*alpha)]
        plt.plot(x, y, 'k-', linewidth=2)



######################################################################
#
#   Test Function
#
def TestPlanarUtilities():
    # Create the figure.
    visual = Visualization()


    # Test points near the triangle.
    print("Testing PointNearTriangle")
    visual.ClearFigure()
    visual.DrawTriangle(triangle)
    for p in pointlist:
        if PointNearTriangle(d, p, triangle):
            visual.DrawPoint(p)
    visual.ShowFigure()
    input("Hit return to continue")


    # Test points near the box.
    print("Testing PointNearBox")
    visual.ClearFigure()
    visual.DrawBox(box)
    for p in pointlist:
        if PointNearBox(d, p, box):
            visual.DrawPoint(p)
    visual.ShowFigure()
    input("Hit return to continue")

    
    # Test segments across multiple angles crossing the segment.
    print("Testing SegmentCrossSegment")
    for (dx, dy) in deltalist:
        visual.ClearFigure()
        visual.DrawSegment(segment)
        for (x,y) in pointlist:
            if SegmentCrossSegment(((x,y), (x+dx,y+dy)), segment):
                visual.DrawPoint((x,y))
        visual.ShowFigure()
    input("Hit return to continue")


    # Test segments across multiple angles near the segment. 
    print("Testing SegmentNearSegment")
    for (dx, dy) in deltalist:
        visual.ClearFigure()
        visual.DrawSegment(segment)
        for (x,y) in pointlist:
            if SegmentNearSegment(d, ((x,y), (x+dx,y+dy)), segment):
                visual.DrawPoint((x,y))
        visual.ShowFigure()
    input("Hit return to continue")


    # Test segments across multiple angles crossing the triangle.
    print("Testing SegmentCrossTriangle")
    for (dx, dy) in deltalist:
        visual.ClearFigure()
        visual.DrawTriangle(triangle)
        for (x,y) in pointlist:
            if SegmentCrossTriangle(((x,y), (x+dx,y+dy)), triangle):
                visual.DrawPoint((x,y))
        visual.ShowFigure()
    input("Hit return to continue")


    # Test segments across multiple angles crossing the box.
    print("Testing SegmentCrossBox")
    for (dx, dy) in deltalist:
        visual.ClearFigure()
        visual.DrawBox(box)
        for (x,y) in pointlist:
            if SegmentCrossBox(((x,y), (x+dx,y+dy)), box):
                visual.DrawPoint((x,y))
        visual.ShowFigure()
    input("Hit return to continue")


    # Test segments across multiple angles crossing the arc.
    print("Testing SegmentCrossArc")
    for (dx, dy) in deltalist:
        visual.ClearFigure()
        visual.DrawArc(arc)
        for (x,y) in pointlist:
            if SegmentCrossArc(((x,y), (x+dx,y+dy)), arc):
                visual.DrawPoint((x,y))
        visual.ShowFigure()
    input("Hit return to continue")


#
#  Main Code
#
if __name__== "__main__":
    TestPlanarUtilities()
