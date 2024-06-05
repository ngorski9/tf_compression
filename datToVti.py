import numpy as np
from pyevtk.hl import *
import struct

def convert(datFile, dims, vtiFile):
    dat = open(datFile, "rb")
    out = np.zeros(dims, dtype=np.float32)

    for k in range(dims[2]):
        for j in range(dims[1]):
            for i in range(dims[0]):
                out[i,j,k] = struct.unpack("<f", dat.read(4))[0]
    
    imageToVTK(vtiFile, pointData={"arr" : out})
    dat.close()

if __name__ == "__main__":
    dims = (1,200,100)
    convert("../output/d.dat", dims, "../output/d")
    convert("../output/r.dat", dims, "../output/r")
    convert("../output/s.dat", dims, "../output/s")
    convert("../output/theta.dat", dims, "../output/theta")