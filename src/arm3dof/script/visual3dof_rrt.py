#!/usr/bin/env python3
#
#   visual.py
#
#
#   Publish:   /joint_states   sensor_msgs/JointState
#
import rospy
import math
import numpy as np

from sensor_msgs.msg   import JointState

from arm3dof_rrt import plan_rrt


#
#  Trajectory Segment Objects
#
#  All these segments are set up to start at time 0, running to some
#  specified end time.  They each store whichever internal parameters
#  they need.  And provide both an evaluate(t) and duration() method.
#  Using numpy, they work with arrays of joints.
#
class CubicSpline:
    # Initialize.
    def __init__(self, p0, v0, pf, vf, T):
        # Precompute the spline parameters.
        self.T = T
        self.a = p0
        self.b = v0
        self.c =  3*(pf-p0)/T**2 - vf/T    - 2*v0/T
        self.d = -2*(pf-p0)/T**3 + vf/T**2 +   v0/T**2

    # Report the segment's duration (time length).
    def duration(self):
        return(self.T)

    # Compute the position/velocity for a given time (w.r.t. t=0 start).
    def evaluate(self, t):
        # Compute and return the position and velocity.
        p = self.a + self.b * t +   self.c * t**2 +   self.d * t**3
        v =          self.b     + 2*self.c * t    + 3*self.d * t**2
        return (p,v)

class Goto(CubicSpline):
    # Use zero initial/final velocities (of same size as positions).
    def __init__(self, p0, pf, T):
        CubicSpline.__init__(self, p0, 0*p0, pf, 0*pf, T)

class Hold(Goto):
    # Use the same initial and final positions.
    def __init__(self, p, T):
        Goto.__init__(self, p, p, T)

class Stay(Hold):
    # Use an infinite time (stay forever).
    def __init__(self, p):
        Hold.__init__(self, p, math.inf)


#
#  Generator Class
#
class Generator:
    # Initialize.
    def __init__(self, path):
        # Create a publisher to send the joint commands.  Add some time
        # for the subscriber to connect.  This isn't necessary, but means
        # we don't start sending messages until someone is listening.
        self.pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        rospy.sleep(0.25)

        self.segments = []
        for i in range(len(path) - 1):
            self.segments.append(Goto(path[i].state.ts, path[i+1].state.ts, 1.0))
        self.segments.append(Hold(path[-1].state.ts, 1))

        # Initialize the current segment index and starting time t0.
        self.index = 0
        self.t0    = 0.0

    # Update is called every 10ms!
    def update(self, t):
        # If the current segment is done, shift to the next.
        dur = self.segments[self.index].duration()
        if (t - self.t0 >= dur):
            self.t0    = (self.t0    + dur)
            self.index = (self.index + 1) % len(self.segments)  # cyclic!

        # Grab the spline position/velocity output for the current time.
        (theta, thetadot) = self.segments[self.index].evaluate(t - self.t0)

        # Create and send the command message.  Note the names have to
        # match the joint names in the URDF.  And their number must be
        # the number of position/velocity elements.
        cmdmsg = JointState()
        cmdmsg.name         = ['theta1', 'theta2', 'theta3']
        cmdmsg.position     = theta
        cmdmsg.velocity     = thetadot
        cmdmsg.header.stamp = rospy.Time.now()
        self.pub.publish(cmdmsg)


#
#  Main Code
#
if __name__ == "__main__":
    # Prepare/initialize this node.
    rospy.init_node('better')

    # Instantiate the trajectory generator object, encapsulating all
    # the computation and local variables.
    # rrt plan
    path = plan_rrt()

    generator = Generator(path)

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))


    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    starttime = rospy.Time.now()
    while not rospy.is_shutdown():

        # Current time (since start)
        servotime = rospy.Time.now()
        t = (servotime - starttime).to_sec()

        # Update the controller.
        generator.update(t)

        # Wait for the next turn.  The timing is determined by the
        # above definition of servo.
        servo.sleep()
