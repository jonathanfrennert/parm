import numpy as np

''' Given a line AB and a point C, return the minimum distance
    between both objects.
    point_A, point_B, point_C each given by [3 x 1] numpy array
'''
def distLineAndPoint(point_A, point_B, point_C):
    # Convert to horizontal
    C = point_C.T
    A = point_A.T
    B = point_B.T
    
    # Unit vector describing AB
    u = (B - A)/np.linalg.norm(B - A)

    # Scalar Projections onto the line
    s = (C - A) @ u.T
    t = (C - B) @ u.T

    # Length of the line.
    L = abs(s - t)

    # If the projection of the point in line space lies:
   
    # Before the first point of the line length
    if s < 0:
        # Distance between C and A
        return np.linalg.norm(C - A)

    # Within the line length
    elif s <= L:
        # Perpendicular distance between C and AB
        return np.linalg.norm((np.cross((C - A), u)))

    # Past the last point of the line length
    else:
        # Distance between C and B
        return np.linalg.norm(C - B)

''' Check whether a line given by points A and B intersects a
    sphere centered at point C. Return True upon intersection
    and False otherwise.
'''
def lineInSphere(point_A, point_B, point_C, radius):
    d = distLineAndPoint(point_A, point_B, point_C)

    if d <= radius:
        return True
    return False
