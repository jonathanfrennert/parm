import numpy as np

# assuming lines is something like: [(p1, p2), ...)]
# boxes = list of bbox objects
def inFreeSpace(lines, boxes):
    for i in range(len(lines)):
        for j in range(len(boxes)):
            # Stack coordinates with an extra 1 for the 4x4 Transformation
            lp1 = np.vstack((lines[i][0],1))
            lp2 = np.vstack((lines[i][1],1))
            if boxes[j].LineInBox(lp1,lp2):
                return False
    # if no lines were in a box, return true
    return True
