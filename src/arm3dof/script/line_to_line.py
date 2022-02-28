import numpy as np

# Reference: https://stackoverflow.com/a/18994296
def safe_distance(delta, r1, r2):
    """
    Determine if the two distance between two lines is safe
    """
    return (delta - (r1 + r2)) > 0

def line_safe_distance(A, B):
    '''
    In this situation we are given two line segments in 3D.
    Line Segment A has endpoints of coordinates a0 and a1 respectively. It also has a radius parameter r_a
    Line Segment B has endpoints of coordinates b0 and b1 respectively. It also has a radius parameter r_b
    We calculate the closest points on each segment and the distance between them.
    After this we determine whether these points are safe distance or not
    '''

    # Calculate denomitator
    a0, a1, r_a = np.array(A[:3]), np.array(A[3:6]), A[6]
    b0, b1, r_b = np.array(B[:3]), np.array(B[3:6]), B[6]

    # Normalize
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))

        # Take overlapping into account
        d1 = np.dot(_A,(b1-a0))

        # Dealing with segment B before A
        if d0 <= 0 >= d1:
            # In the case when B is before A, we just take euclidean norm
            if np.absolute(d0) < np.absolute(d1):
                distance = np.linalg.norm(a0 - b0)
                return safe_distance(distance, r_a, r_b)
            # Do norm between a0 and b1
            distance = np.linalg.norm(a0 - b1)
            return safe_distance(distance, r_a, r_b)

        # Dealing with segment B after A
        elif d0 >= magA <= d1:
            # In the case when B is after A
            if np.absolute(d0) < np.absolute(d1):
                distance = np.linalg.norm(a1 - b0)
                return safe_distance(distance, r_a, r_b)

            distance = np.linalg.norm(a1-b1)
            return safe_distance(distance, r_a, r_b)

        # Segments overlap
        distance = np.linalg.norm(((d0*_A)+a0)-b0)
        return safe_distance(distance, r_a, r_b)

    # Lines overlap: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    # Projected closest point on segment A
    pA = a0 + (_A * t0)

    # Projected closest point on segment B
    pB = b0 + (_B * t1)

    # projections on A
    if t0 < 0:
        pA = a0
    elif t0 > magA:
        pA = a1

    # projections on B
    if t1 < 0:
        pB = b0
    elif t1 > magB:
        pB = b1

    # projection of line segment A
    if t0 < 0 or t0 > magA:
        dot = np.dot(_B,(pA-b0))
        if dot < 0:
            dot = 0
        elif dot > magB:
            dot = magB
        pB = b0 + (_B * dot)

    # projection of line segment B
    if (t1 < 0) or (t1 > magB):
        dot = np.dot(_A,(pB-a0))
        if dot < 0:
            dot = 0
        elif dot > magA:
            dot = magA
        pA = a0 + (_A * dot)

    # Calculate distance in case of overlap in lines
    distance = np.linalg.norm(pA - pB)
    return safe_distance(distance, r_a, r_b)