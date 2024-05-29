if __name__ == "__main__":
    import numpy as np
    from math import sin, cos, pi

    def eigs(theta):
        matrix = np.array([[ cos(theta), sin(theta) ], [sin(theta), -cos(theta)]] )
        print(np.linalg.eig(matrix)[1])

    def mat2(theta):
        return np.array( [ [cos(theta), -sin(theta)], [sin(theta), -cos(theta)] ] )
    
    matrix = mat2(pi/2)
    print(matrix)
    print(np.linalg.eig(matrix)[1])