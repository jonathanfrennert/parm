import numpy as np

''' http://www.3dkingdoms.com/weekly/weekly.php?a=21
'''
class bbox:
    def __init__(self, T, extents):
        # transformation matrix
        self.T = T;
        # half lengths of box along each axis
        self.extents = extents; 
    
    def getSize(self):
        return self.extents
    
    def getCenterPoint(self):
        return

    # lp1 and lp2 refers to end points of a line
    # needs to be 4x1 with the last entry a 1
    def LineInBox(self, lp1, lp2):
        # Get the inverse transformation
        R = self.T[0:3, 0:3]
        Tinv = np.eye(4)
        Tinv[0:3,0:3] = R.T
        Tinv[0:3,3:4] = -R.T @ self.T[0:3,3:4]
        Tinv[3:4,3:4] = 1;

        # Put the line in box coordinate frame
        lb1 = Tinv @ lp1
        lb2 = Tinv @ lp2

        lbmid = .5*(lb1 + lb2)
        L = lb1 - lbmid
        lext = abs(L)

        # Using the Separating Axis Theorem
        if (abs(lbmid[0]) > self.extents[0] + lext[0]): return False
        if (abs(lbmid[1]) > self.extents[1] + lext[1]): return False
        if (abs(lbmid[2]) > self.extents[2] + lext[2]): return False

        # Cross products of lines and axes
        if ( abs( lbmid[1] * L[2] - lbmid[2] * L[1])  >  (self.extents[1] * lext[2] + self.extents[2] * lext[1]) ): return False;
        if ( abs( lbmid[0] * L[2] - lbmid[2] * L[0])  >  (self.extents[0] * lext[2] + self.extents[2] * lext[0]) ): return False;
        if ( abs( lbmid[0] * L[1] - lbmid[1] * L[0])  >  (self.extents[0] * lext[1] + self.extents[1] * lext[0]) ): return False;
        
        # No separating axis, the line intersects
        return True