import os
import struct

if __name__ == "__main__":

    inf = "../data/sym/brain23test"
    outf = "../data/sym/brain23altered"

    dims = (66,108,108)

    in1 = open(f"{inf}/row_1_col_1.dat", "rb")
    bytes1 = in1.read()
    in1.close()

    in2 = open(f"{inf}/row_1_col_2.dat","rb")
    bytes2 = in2.read()
    in2.close()

    in3 = open(f"{inf}/row_2_col_2.dat","rb")
    bytes3 = in3.read()
    in3.close()

    if not os.path.isdir(outf):
        os.mkdir(outf)

    floats1 = struct.unpack(f"<{dims[0]*dims[1]*dims[2]}d", bytes1)
    floats2 = struct.unpack(f"<{dims[0]*dims[1]*dims[2]}d", bytes2)
    floats3 = struct.unpack(f"<{dims[0]*dims[1]*dims[2]}d", bytes3)

    outfloats1 = []
    outfloats2 = []
    outfloats3 = []

    slice_size = dims[0]*dims[1]

    num_erased = 0

    out1 = open(f"{outf}/row_1_col_1.dat", "wb")
    out2 = open(f"{outf}/row_1_col_2.dat", "wb")
    out3 = open(f"{outf}/row_2_col_2.dat", "wb")

    for slice in range(dims[2]):
        testfloats1 = floats1[slice*slice_size:(slice+1)*slice_size]
        testfloats2 = floats2[slice*slice_size:(slice+1)*slice_size]
        testfloats3 = floats3[slice*slice_size:(slice+1)*slice_size]

        keep = False

        for i in range(slice_size):
            if testfloats1[i] != 0.0 or testfloats2[i] != 0.0 or testfloats3[i] != 0.0:
                keep = True
        
        if keep:
            out1.write(struct.pack(f"<{slice_size}d", *testfloats1))
            out2.write(struct.pack(f"<{slice_size}d", *testfloats2))
            out3.write(struct.pack(f"<{slice_size}d", *testfloats3))
        else:
            num_erased += 1
    
    print(f"erased {num_erased} slices")