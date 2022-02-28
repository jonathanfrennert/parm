import numpy as np

def distLineSegments(A1, B1, A2, B2):
    p1 = A1.T
    p2 = B1.T
    q1 = A2.T
    q2 = B2.T

    P = p2 - p1
    Q = q2 - q1
    LP = np.linalg.norm(P)
    LQ = np.linalg.norm(Q)

    t = p1 - q1
    PdotQ = (P @ Q.T)[0][0]
    PdotP = (P @ P.T)[0][0]
    QdotQ = (Q @ Q.T)[0][0]
    tdotP = (t @ P.T)[0][0]
    tdotQ = (t @ Q.T)[0][0]

    A = np.array([[PdotP, -PdotQ],[PdotQ, -QdotQ]]);
    b = np.array([-tdotP, -tdotQ])

    x = np.linalg.solve(A,b)
    A = p1 + x[0] * P
    B = q1 + x[1] * Q
    
    
    priorP = x[0] < 0
    postP = x[0] > LP
    priorQ = x[1] < 0
    postQ = x[1] > LQ

    onP = not (priorP or postP)
    onQ = not (priorQ or postQ)

    # Parallel case:
    
    # Skewer but projected segments overlap:
    if onP and onQ:
        return np.linalg.norm(A-B)

    if onP and not onQ:
        if priorQ:
            return np.linalg.norm(A-q1)
        else: #postQ
            return np.linalg.norm(A-q2)

    if not onP and onQ:
        if priorP:
            return np.linalg.norm(B-p1)
        else: #postP
            return np.linalg.norm(B-p2)

    # Skewer but projected segments do not overlap:
    # if not onP and not onQ:
        # distLineAndPoint()

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



def main():
    p1 = np.array([-1, 0, 0]).reshape(3,1)
    p2 = np.array([ 1, 0, 0]).reshape(3,1)
    q1 = np.array([0, -1, 1]).reshape(3,1)
    q2 = np.array([0, 1, 1]).reshape(3,1)
    distLineSegments(p1, p2, q1, q2)

if __name__ == '__main__':
    main()