import numpy as np

def fkin(ls, ts, cs, ss):
    """
    Give forward kinematics of each joint
    """
    # TODO: Make it a loop
    p1 = np.array([0, 0, ls[0]])

    # Get r, z
    r = ls[1] * cs[1]
    z = ls[1] * ss[1]
    p2 = np.array([- r * ss[0], r * cs[0], z])

    # Get r, l for third bar
    r += ls[2] * np.cos(ts[1] + ts[2])
    z += ls[2] * np.sin(ts[1] + ts[2])

    # new joint
    p3 = np.array([- r * ss[0], r * cs[0], z])

    # Get r, l for foruth bar
    r += ls[3] * np.cos(ts[1] + ts[2] + ts[3])
    z += ls[3] * np.sin(ts[1] + ts[2] + ts[3])

    # position update
    p4 = np.array([-r * ss[0], r * cs[0], z])

    # Get r,l for final bar
    r += ls[4] * np.cos(ts[1] + ts[2] + ts[3] + ts[4])
    z += ls[4] * np.sin(ts[1] + ts[2] + ts[3] + ts[4])

    # position update
    p5 = np.array([-r * ss[0], r * cs[0], z])

    return np.array([p1, p2, p3, p4, p5])
