#!/usr/bin/env python3
#
#   prmcar.py
#
import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from sklearn.neighbors import KDTree
from planarutils import *


######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#
(wroad)                  = 6                            # Road
(xspace, lspace, wspace) = (5, 6, 2.5)                  # Parking Space
(xmin, ymin, xmax, ymax) = (0, 0, 20, wroad+wspace)     # Overall boundary

# Define the car.  Outline and center of rotation.
(lcar, wcar) = (4, 2)           # Length/width
lb           = 0.5              # Center of rotation to back bumper
lf           = lcar - lb        # Center of rotation to front bumper
wc           = wcar/2           # Center of rotation to left/right
wheelbase    = 3                # Center of rotation to front wheels

# Max steering angle.
steermax    = np.pi/4
tansteermax = np.tan(steermax)

# Construct the walls.
walls = (((xmin         , ymin        ), (xmax         , ymin        )),
         ((xmax         , ymin        ), (xmax         , wroad       )),
         ((xmax         , wroad       ), (xspace+lspace, wroad       )),
         ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
         ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
         ((xspace       , wroad+wspace), (xspace       , wroad       )),
         ((xspace       , wroad       ), (xmin         , wroad       )),
         ((xmin         , wroad       ), (xmin         , 0           )))

# Pick your start and goal locations.
(startx, starty, startt) = (2.0, 2.0, 0.0)
(goalx,  goaly,  goalt)  = (xspace + (lspace-lcar)/2 + lb, wroad + wc, 0.0)

# Spacing for drawing/testing
ddraw = 0.5

# PRM parameters
N = 800     # 200
K = 50      # 40


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

        # Create a new axes, enable the grid, and prepare the axes.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim( -9, 15)
        plt.gca().set_ylim(-12, 12)
        plt.gca().set_aspect('equal')

    def ShowParkingSpot(self):
        # Define the region (axis limits).
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)

        # Show the walls.
        for wall in walls:
            plt.plot([wall[0][0], wall[1][0]],
                     [wall[0][1], wall[1][1]], 'k', linewidth=2)

        # Mark the locations.
        plt.gca().set_xticks(list(set([wall[0][0] for wall in walls])))
        plt.gca().set_yticks(list(set([wall[0][1] for wall in walls])))

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)


######################################################################
#
#   State Definition
#
# Angular distance within +/- 180deg.
def AngleDiff(t1, t2):
    return (t1-t2) - 2.0*np.pi * round(0.5*(t1-t2)/np.pi)

#
#   State = One set of coordinates
#
class State:
    def __init__(self, x, y, theta):
        # Pre-compute the trigonometry.
        s = np.sin(theta)
        c = np.cos(theta)

        # Remember the state (x,y,theta).
        self.x = x
        self.y = y
        self.t = theta
        self.s = s
        self.c = c

        # Box (4 corners: frontleft, backleft, backright, frontright)
        self.box = ((x + c*lf - s*wc, y + s*lf + c*wc),
                    (x - c*lb - s*wc, y - s*lb + c*wc),
                    (x - c*lb + s*wc, y - s*lb - c*wc),
                    (x + c*lf + s*wc, y + s*lf - c*wc))

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<XY %5.2f,%5.2f @ %5.1f deg>" %
                (self.x, self.y, self.t * (180.0/np.pi)))

    # Draw the state.
    def Draw(self, fig, color, **kwargs):
        b = self.box
        # Box
        plt.plot((b[0][0], b[1][0]), (b[0][1], b[1][1]), color, **kwargs)
        plt.plot((b[1][0], b[2][0]), (b[1][1], b[2][1]), color, **kwargs)
        plt.plot((b[2][0], b[3][0]), (b[2][1], b[3][1]), color, **kwargs)
        plt.plot((b[3][0], b[0][0]), (b[3][1], b[0][1]), color, **kwargs)
        # Headlights
        plt.plot(0.9*b[3][0]+0.1*b[0][0], 0.9*b[3][1]+0.1*b[0][1], color+'o')
        plt.plot(0.1*b[3][0]+0.9*b[0][0], 0.1*b[3][1]+0.9*b[0][1], color+'o')

    # Return a tuple of the coordinates for KDTree.
    def Coordinates(self):
        return (self.x, self.y, wheelbase*self.s, wheelbase*self.c)


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


######################################################################
#
#   Local Planner
#
#   There are many options here.  We assume we drive at a constant
#   speed and for a given distance at one steering angle (turning
#   radius), followed by another distance at the opposite steering
#   angle.  As such, we also define an arc, being a constant speed and
#   steering angle.
#
#   Note the tan(steeringAngle) = wheelBase / turningRadius
#
class Arc:
    def __init__(self, fromState, toState, distance, tansteer):
        # Remember the parameters.
        self.fromState = fromState
        self.toState   = toState
        self.distance  = distance       # can be nagative when backing up!
        self.tansteer  = tansteer       # pos = turn left, neg = turn right

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Arc %s to %s, distance %5.2f m, steer %5.2f deg>" %
                (self.fromState, self.toState, self.distance,
                 np.rad2deg(np.arctan(self.tansteer))))

    # Return the absolute length.
    def Length(self):
        return abs(self.distance)

    # Return an intermediate state, d along arc.  Note we take care
    # never to divide, so this works for both arcs and straight lines.
    def IntermediateState(self, d):
        if self.tansteer == 0:
            return State(self.fromState.x + self.fromState.c * d,
                         self.fromState.y + self.fromState.s * d,
                         self.fromState.t)
        else:
            r   = wheelbase / self.tansteer
            phi = d / r 
            ds  = np.sin(self.fromState.t + phi) - self.fromState.s
            dc  = np.cos(self.fromState.t + phi) - self.fromState.c
            return State(self.fromState.x + r * ds,
                         self.fromState.y - r * dc,
                         self.fromState.t + phi)

    # Draw the arc (showing the intermediate states).
    def Draw(self, fig, color, **kwargs):
        n = max(2, int(np.ceil(abs(self.distance) / ddraw)))
        d = self.distance / n
        for i in range(1,n):
            self.IntermediateState(d*i).Draw(fig, color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether in free space, with given spacing.
    def Valid(self):
        # First check the steering angle (inv turning radius).
        if abs(self.tansteer) > tansteermax:
            return False

        # If straight: Check back left/right corners segments.
        if self.tansteer == 0:
            seg1 = (self.fromState.box[1], self.toState.box[1])
            seg2 = (self.fromState.box[2], self.toState.box[2])
            for wall in walls:
                if (SegmentCrossSegment(wall, seg1) or
                    SegmentCrossSegment(wall, seg2)):
                    return False
            return True

        # Compute the center of rotation.
        r  = wheelbase / self.tansteer
        pc = (self.fromState.x - r * self.fromState.s,
              self.fromState.y + r * self.fromState.c)

        # Check the arcs based on turning left/right and direction:
        if self.tansteer > 0:
            # Turning left:  Check min (backleft) and max (frontright) radius
            if self.distance > 0:
                arc1 = (pc, self.fromState.box[1], self.toState.box[1])
                arc2 = (pc, self.fromState.box[3], self.toState.box[3])
            else:
                arc1 = (pc, self.toState.box[1], self.fromState.box[1])
                arc2 = (pc, self.toState.box[3], self.fromState.box[3])
        else:
            # Turning right: Check min (backright) and max (frontleft) radius
            if self.distance > 0:
                arc1 = (pc, self.toState.box[2], self.fromState.box[2])
                arc2 = (pc, self.toState.box[0], self.fromState.box[0])
            else:
                arc1 = (pc, self.fromState.box[2], self.toState.box[2])
                arc2 = (pc, self.fromState.box[0], self.toState.box[0])

        for wall in walls:
            if (SegmentCrossArc(wall, arc1) or
                SegmentCrossArc(wall, arc2)):
                return False
        return True

#
#   Local Plan.  I'm using two arcs, but please feel free to create
#   whatever local planner you prefer.
#
class LocalPlan:
    def __init__(self, fromState, toState):
        # Compute the connection.
        (midState, arc1, arc2) = self.ComputeConnection(fromState, toState)

        # Save the information.
        self.fromState = fromState
        self.midState  = midState
        self.toState   = toState
        self.arc1      = arc1
        self.arc2      = arc2

    def ComputeConnection(self, fromState, toState):
        # Grab the starting and final coordinates.
        (x1, x2) = (fromState.x, toState.x)
        (y1, y2) = (fromState.y, toState.y)
        (t1, t2) = (fromState.t, toState.t)
        (s1, s2) = (fromState.s, toState.s)
        (c1, c2) = (fromState.c, toState.c)

        # COMPUTATION.  I find it useful to compute an turning radius
        # (or inverse thereof).  It is very useful to allow the local
        # planner to back up!!  Along one arc, or along all arcs...
        # You should know
        wb = wheelbase

        ....TODO....
    
        # Check for zero steering (infinite radius).
        if invR == 0:
            # Straight line!
            tm = t1                     # Theta at mid point
            xm = 0.5*(x1+x2)            # X coordinate at mid point
            ym = 0.5*(y1+y2)            # Y coordinate at mid point
            d1 = 0.5*np.sqrt(a)         # Distance on first arc
            d2 = d1                     # Distance on second arc

        else:
            # Else use two arcs...  Note it may help to back up!!
            tm = 
            xm = 
            ym = 
            d1 = 
            d2 = 

        # Return the mid state and two arcs.  Again, you may choose
        # differently, but the below is my approach.
        .... CHECK THIS - DOES  IT MATCH YOUR APPROACH ...
        tansteer = invR * wheelbase
        midState = State(xm, ym, tm)
        arc1     = Arc(fromState, midState, d1,  tansteer)
        arc2     = Arc(midState , toState,  d2, -tansteer)
        return (midState, arc1, arc2)

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Arc1 %s\n Arc2 %s\n => Length %5.2f>" %
                (self.arc1, self.arc2, self.Length()))

    # Return the absolute length.
    def Length(self):
        return self.arc1.Length() + self.arc2.Length()

    # Draw the local plan (showing the mid state and two arcs).
    def Draw(self, fig, color, **kwargs):
        self.midState.Draw(fig, color, **kwargs)
        self.arc1.Draw(fig, color, **kwargs)
        self.arc2.Draw(fig, color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether the midpoint is in free space and
    # both arcs are valid (turning radius and collisions).
    def Valid(self):
        return (self.midState.InFreeSpace() and
                self.arc1.Valid() and
                self.arc2.Valid())


######################################################################
#
#   A* Functions
#
#
#   Node class upon which to build the graph (roadmap) and which
#   supports the A* search tree.
#
class Node:
    def __init__(self, state):
        # Save the state matching this node.
        self.state = state

        # Edges used for the graph structure (roadmap).
        self.childrenandcosts = []
        self.parents          = []

        # Status, edge, and costs for the A* search tree.
        self.seen        = False
        self.done        = False
        self.treeparent  = []
        self.costToReach = 0
        self.costToGoEst = np.inf
        self.cost        = self.costToReach + self.costToGoEst

    # Define the "less-than" to enable sorting by cost in A*.
    def __lt__(self, other):
        return self.cost < other.cost

    # Distance to another node, for A*, using the state distance.
    def Distance(self, other):
        return self.state.Distance(other.state)


#
#   A* Planning Algorithm
#
def AStar(nodeList, start, goal):
    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Clear the search tree (for repeated searches).
    for node in nodeList:
        node.seen = False
        node.done = False

    # Begin with the start state on-deck.
    start.done        = False
    start.seen        = True
    start.treeparent  = None
    start.costToReach = 0
    start.costToGoEst = start.Distance(goal)
    start.cost        = start.costToReach + start.costToGoEst
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    while True:
        # Grab the next node (first on deck).
        node = onDeck.pop(0)

        # Add the children to the on-deck queue (or update)
        for (child,tripcost) in node.childrenandcosts:
            # Skip if already done.
            if child.done:
                continue

            # Compute the cost to reach the child via this new path.
            costToReach = node.costToReach + tripcost

            # Just add to on-deck if not yet seen (in correct order).
            if not child.seen:
                child.seen        = True
                child.treeparent  = node
                child.costToReach = costToReach
                child.costToGoEst = child.Distance(goal)
                child.cost        = child.costToReach + child.costToGoEst
                bisect.insort(onDeck, child)
                continue

            # Skip if the previous cost was better!
            if child.costToReach <= costToReach:
                continue

            # Update the child's connection and resort the on-deck queue.
            child.treeparent  = node
            child.costToReach = costToReach
            child.cost        = child.costToReach + child.costToGoEst
            onDeck.remove(child)
            bisect.insort(onDeck, child)

        # Declare this node done.
        node.done = True

        # Check whether we have processed the goal (now done).
        if (goal.done):
            break

        # Also make sure we still have something to look at!
        if not (len(onDeck) > 0):
            return []

    # Build the path.
    path = [goal]
    while path[0].treeparent is not None:
        path.insert(0, path[0].treeparent)

    # Return the path.
    return path


######################################################################
#
#   PRM Functions
#
#
# Sample the space
#
def AddNodesToList(nodeList, N):
    # ...TODO... YOU MAY WANT TO ADD/ALTER THE SAMPLING!!!

    # Add uniformly distributed samples over the entire space.
    while (len(nodeList) < N):
        state = State(random.uniform(xmin, xmax),
                      random.uniform(ymin, ymax),
                      random.uniform(-np.pi/4, np.pi/4))
        if state.InFreeSpace():
            nodeList.append(Node(state))


#
#   Connect the nearest neighbors
#
def ConnectNearestNeighbors(nodeList, K):
    # Clear any existing neighbors.
    for node in nodeList:
        node.childrenandcosts = []
        node.parents          = []

    # Determine the indices for the nearest neighbors.  This also
    # reports the node itself as the closest neighbor, so add one
    # extra here and ignore the first element below.
    X   = np.array([node.state.Coordinates() for node in nodeList])
    kdt = KDTree(X)
    idx = kdt.query(X, k=(K+1), return_distance=False)

    # Add the edges (from parent to child).  Ignore the first neighbor
    # being itself.
    for i, nbrs in enumerate(idx):
        print(i)
        children = [child for (child,_) in nodeList[i].childrenandcosts]
        for n in nbrs[1:]:
            if not nodeList[n] in children:
                plan = LocalPlan(nodeList[i].state, nodeList[n].state)
                if plan.Valid():
                    cost = plan.Length()
                    nodeList[i].childrenandcosts.append((nodeList[n], cost))
                    nodeList[n].childrenandcosts.append((nodeList[i], cost))
                    nodeList[n].parents.append(nodeList[i])
                    nodeList[i].parents.append(nodeList[n])


#
#  Post Process the Path
#
def UniqueStates(path):
    # Initialize the state list.
    states = [path[0].state]

    # Add all arc points (if they are unique).
    for i in range(1, len(path)):
        plan = LocalPlan(path[i-1].state, path[i].state)
        if (states[-1].Distance(plan.midState) > 0.01):
            states.append(plan.midState)
        if (states[-1].Distance(plan.toState) > 0.01):
            states.append(plan.toState)

    # Return the full list.
    return states
    
def PostProcess(path):
    # Grab all states, including the intermediate states between arcs.
    states = UniqueStates(path)
    
    # Check whether we can skip states.
    # .... TODO ... is there a shorter list that still works?

    # Rebuild and return the path (list of nodes).
    return [Node(state) for state in states]


######################################################################
#
#  Main Code
#
def CheckLocalPlan(fig, fromState, toState):
    # Clear the figure.
    fig.ClearFigure()

    # Show the initial and final states.
    fromState.Draw(fig, 'r', linewidth=2)
    toState.Draw(fig,   'r', linewidth=2)

    # Compute and show the local plan.
    plan = LocalPlan(fromState, toState)
    plan.Draw(fig, 'b', linewidth=1)

    # Show/report
    fig.ShowFigure()
    print("Local plan from %s to %s" % (fromState, toState))
    input("(hit return to continue)")

def TestLocalPlanner(fig):
    CheckLocalPlan(fig, State(0, 0, 0),       State(8,  0, 0))
    CheckLocalPlan(fig, State(0, 0, 0),       State(8,  8, np.pi/2))
    CheckLocalPlan(fig, State(0, 0, 0),       State(8,  8, 0))
    CheckLocalPlan(fig, State(0, 0, 0),       State(8, -8, 0))
    CheckLocalPlan(fig, State(0, 0, 0),       State(8,  0, -np.pi/4))

    CheckLocalPlan(fig, State(0, 0, 0),       State(0,  8, 0))
    CheckLocalPlan(fig, State(0, 0, 0),       State(0,  8, np.pi/2))

    CheckLocalPlan(fig, State(0, 0, 0),       State(6, -6,   np.pi/2))
    CheckLocalPlan(fig, State(0, 0, 0),       State(8, -8,   np.pi/2))


def DrawPath(path, fig, color, **kwargs):
    # Draw the individual local plans
    for i in range(len(path)-1):
        plan = LocalPlan(path[i].state, path[i+1].state)
        plan.Draw(fig, color, **kwargs)

    # Print the unique path elements.
    print("Unique steps in path:")
    for (i, state) in enumerate(UniqueStates(path)):
        print(i, state)

    
def main():
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')

    # Create the figure.
    fig = Visualization()

    # Test the local planner:
    TestLocalPlanner(fig)

    # Switch to the road figure.
    fig.ClearFigure()
    fig.ShowParkingSpot()
    
        
    # Create the start/goal nodes.
    startnode = Node(State(startx, starty, startt))
    goalnode  = Node(State(goalx,  goaly,  goalt))

    # Show the start/goal states.
    startnode.state.Draw(fig, 'r', linewidth=2)
    goalnode.state.Draw(fig,  'r', linewidth=2)
    fig.ShowFigure()
    input("Showing basic world (hit return to continue)")


    # Create the list of sample points.
    start = time.time()
    nodeList = []
    AddNodesToList(nodeList, N)
    print('Sampling took ', time.time() - start)

    # # Show the sample states.
    # for node in nodeList:
    #     node.state.Draw(fig, 'k', linewidth=1)
    # fig.ShowFigure()
    # input("Showing the nodes (hit return to continue)")

    # Add the start/goal nodes.
    nodeList.append(startnode)
    nodeList.append(goalnode)


    # Connect to the nearest neighbors.
    start = time.time()
    ConnectNearestNeighbors(nodeList, K)
    print('Connecting took ', time.time() - start)

    # # Show the neighbor connections.
    # for node in nodeList:
    #     for child in node.children:
    #         plan = LocalPlan(node.state, child.state)
    #         plan.Draw(fig, 'g-', linewidth=0.5)
    # fig.ShowFigure()
    # input("Showing the full graph (hit return to continue)")


    # Run the A* planner.
    start = time.time()
    path = AStar(nodeList, startnode, goalnode)
    print('A* took ', time.time() - start)
    if not path:
        print("UNABLE TO FIND A PATH")
        return


    # Show the path.
    DrawPath(path, fig, 'r', linewidth=1)
    fig.ShowFigure()
    input("Showing the raw path (hit return to continue)")

    # Post Process the path.
    path = PostProcess(path)

    # Show the post-processed path.
    DrawPath(path, fig, 'b', linewidth=2)
    fig.ShowFigure()
    input("Showing the post-processed path (hit return to continue)")


if __name__== "__main__":
    main()
