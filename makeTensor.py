import struct
import numpy as np
import os
from math import sin, cos

if __name__ == "__main__":

    # the dreaded ellipse blob :o
    # d = [1.6, 1.3, 1.4, 2.0]
    # r = [0.3, 1.0, 4.2, 3.0]
    # s = [4.7, 3.6, 8.8, 4.0]
    # t = [4.27, 6.28, 2.2, 0.5]

    # strange multi-overlaps:
    # d = [0.4, 0.6, 0.4, 2.0]
    # r = [2.77, 1.0, -8.2, 3.0]
    # s = [2.77, 2.0, 1.4, 0.5]
    # t = [3.37, 6.28, 0.7, 0.5]

    # near miss (two hyperbola situation)
    # d = [0.4, -0.8, 0.8, 2.0]
    # r = [2.77, 1.0, -8.2, 3.0]
    # s = [2.7, 2.0, 1.4, 4.0]
    # t = [3.37, 6.28, 0.7, 0.5]

    # all five colors in one cell.
    # d = [3.4, -3.2, 0.8, 2.0]
    # r = [2.77, 0.4, -8.2, 3.0]
    # s = [2.7, 2.0, 1.4, 4.0]
    # t = [3.37, 6.28, 0.7, 0.5]

    # six disjoint regions.
    # d = [8.6, -2.4, -8.1, 2.0]
    # r = [2.77, 0.1, -8.2, 3.0]
    # s = [5.84, 2.0, 1.4, 4.0]
    # t = [3.37, 6.28, 0.7, 0.5]

    d = [3.6, -4.0, 1.0, 2.0]
    r = [6.3, -2.8, 2.3, 3.0]
    s = [4.27, 3.6, 4.7, 4.0]
    t = [4.5, 5.27, 1.5, 0.5]

    a = []
    b = []
    c = []
    d_ = []

    for i in range(4):
        mat = d[i] * np.array([[1, 0], [0, 1]]) + r[i] * np.array([[0, -1], [1, 0]]) + s[i] * np.array( [[cos(t[i]), sin(t[i])], [sin(t[i]), -cos(t[i])]] )
        a.append(mat[0,0])
        b.append(mat[0,1])
        c.append(mat[1,0])
        d_.append(mat[1,1])

    if not os.path.isdir("../output/test"):
        os.mkdir("../output/test")

    outa = open("../output/test/row_1_col_1.dat", "wb")
    outa.write(struct.pack("<4d", *a))
    outa.close()

    outb = open("../output/test/row_1_col_2.dat", "wb")
    outb.write(struct.pack("<4d", *b))
    outb.close()

    outc = open("../output/test/row_2_col_1.dat", "wb")
    outc.write(struct.pack("<4d", *c))
    outc.close()

    outd = open("../output/test/row_2_col_2.dat", "wb")
    outd.write(struct.pack("<4d", *d_))
    outd.close()