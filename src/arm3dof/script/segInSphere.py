import numpy as np

''' Given a line AB and a point C, return the minimum distance
    between both objects.
    point_A, point_B, point_C each given by [1 x 3] numpy array
'''
def distLineAndPoint(point_A, point_B, point_C):
    C = point_C
    A = point_A
    B = point_B
    
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
    sphere = [x y z R]
    segment = [x1 y1 z1 x2 y2 z2 r]
'''
def segInSphere(segment, sphere):
    d = distLineAndPoint(segment[0:3], segment[3:6], sphere[0:3])

    # distance between sphere and joint minimized at the sum of their sizes
    if d <= sphere[-1] + segment[-1] :
        return True
    return False
