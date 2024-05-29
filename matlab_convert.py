if __name__ == "__main__":
    import scipy.io
    import numpy as np
    import os
    import struct
    import math

    for i in range(1,9):
        os.system(f"mkdir ../data/3d/tf_{i}")
        mat = scipy.io.loadmat(f"../data/3d_raw/tensorfield_{i}.mat")
        print(mat["Sij"][0,1].shape)
        for row in range(3):
            for col in range(3):
                vals = np.real(mat["Sij"][row,col].reshape(-1))
                print(vals.dtype)
                out = open(f"../data/3d/tf_{i}/row_{row+1}_col_{col+1}.dat", "wb")
                if vals.dtype == np.float64:
                    out.write(struct.pack(f"<{len(vals)}d", *list(vals)))
                else:
                    out.write(struct.pack(f"<{len(vals)}f"), *list(vals))
                out.close()
                