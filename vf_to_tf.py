from paraview.simple import *
import numpy as np
import struct
import os

def floatListToDat(li, name, doublePrecision = True):
    out = open(name, "wb")
    if doublePrecision:
        out.write(struct.pack(f"<{len(li)}d", *li))
    else:
        out.write(struct.pack(f"<{len(li)}f", *li))
    out.close()

def vf_gradient(filename, arrayName, outputFolder):
    paraviewVti = XMLImageDataReader(registrationName='source', FileName=[filename])
    paraviewVti.PointArrayStatus = [arrayName]
    paraviewVti.TimeArray = 'None'

    vtkVti = servermanager.Fetch(paraviewVti)
    numPoints = vtkVti.GetNumberOfPoints()

    row_1_col_1 = [0.0] * numPoints
    row_1_col_2 = [0.0] * numPoints
    row_2_col_1 = [0.0] * numPoints
    row_2_col_2 = [0.0] * numPoints

    x,y,_ = vtkVti.GetDimensions()

    arr = vtkVti.GetPointData().GetArray(arrayName)

    index = -1
    for j in range(y):
        for i in range(x):
            index += 1
            val = arr.GetTuple(index)

            dudx = 0
            dvdx = 0

            if i != 0:
                neighborVal = arr.GetTuple(index-1)
                dudx += val[0]-neighborVal[0]
                dvdx += val[1]-neighborVal[1]
            if i != x-1:
                neighborVal = arr.GetTuple(index+1)
                dudx += neighborVal[0]-val[0]
                dvdx += neighborVal[1]-val[1]
            if i != 0 and i != x-1:
                dudx /= 2
                dvdx /= 2
            
            dudy = 0
            dvdy = 0

            if j != 0:
                neighborVal = arr.GetTuple(index-x)
                dudy += val[0]-neighborVal[0]
                dvdy += val[1]-neighborVal[1]
            if j != y-1:
                neighborVal = arr.GetTuple(index+x)
                dudy += neighborVal[0]-val[0]
                dvdy += neighborVal[1]-val[1]
            if j != 0 and j != y-1:
                dudy /= 2
                dvdy /= 2
            
            row_1_col_1[index] = dudx
            row_1_col_2[index] = dudy
            row_2_col_1[index] = dvdx
            row_2_col_2[index] = dvdy
    
    os.system(f"mkdir {outputFolder}")
    floatListToDat(row_1_col_1, f"{outputFolder}/row_1_col_1.dat")
    floatListToDat(row_1_col_2, f"{outputFolder}/row_1_col_2.dat")
    floatListToDat(row_2_col_1, f"{outputFolder}/row_2_col_1.dat")
    floatListToDat(row_2_col_2, f"{outputFolder}/row_2_col_2.dat")

if __name__ == "__main__":
    vf_gradient("../data/vf/dampedOscillator.vti", "Vectors_", "../data/2d/dampedOscillator")