import numpy as np

def fkin(theta, length):
    """
    Give forward kinematics of each join
    """
    # Get length
    l1, l2, l3 = length[0], length[1], length[2]

    # Get angles
    theta1, theta2, theta3 = theta[0], theta[1], theta[2]

    # TODO: Change in future
    p1 = np.array([0, 0, l1])

    # Get r, z
    r = l2 * np.cos(theta2)
    z = l2 * np.sin(theta2)
    p2 = np.array([- r * np.sin(theta1), r * np.cos(theta1) , z])

    # Get r, l for third bar
    r += l3 * np.cos(theta2 + theta3)
    z += l3 * np.sin(theta2 + theta3)

    # End effector
    p3 = np.array([-r * np.sin(theta1), r * np.cos(theta1), z])

    return [p1, p2, p3]