"""
An Example Script Illustrating how to find the geodesic for an isotropic Riemannian
manifold with metric coefficient exp(-<n,x>) where n is a constant vector.
"""
from math import exp
from os import cpu_count

import numpy as np
from geodesic.geometry import Curve
from geodesic.curve_shorten import compute_geodesic


def main():
    # Set dimension of the problem
    dimension = 4

    # Set parameters for computation
    number_of_global_nodes = 16
    number_of_local_nodes = 8
    maximum_average_node_movement = 0.001
    number_of_cpu = cpu_count()

    # Create start and end point NumPy arrays
    start_point = np.zeros(dimension)
    start_point[0] = -1

    end_point = np.zeros(dimension)
    end_point[0] = 1

    # Create constant vector n
    alpha = 0.65
    n = alpha*np.ones(dimension)
    n[0] = 0


    # Define function to describe metric coefficient
    def metric_coefficient(x):
        return exp(-np.inner(n, x))


    def metric_coefficient_gradient(x):
        return -n * exp(-np.inner(n, x))


    print('Starting Example Calculation...')

    # Create curve object for calculation
    curve = Curve(start_point, end_point, number_of_global_nodes)

    # Apply curve shortening procedure to minimise length
    compute_geodesic(curve, number_of_local_nodes, maximum_average_node_movement, metric_coefficient,
                     metric_coefficient_gradient, number_of_cpu)

    # Print shortened curve points
    # Expected output:
    # [[-1.          0.          0.          0.        ]
    #  [-0.90922598  0.07855664  0.07855664  0.07855664]
    #  [-0.80460516  0.15156238  0.15156238  0.15156238]
    #  [-0.68508424  0.21649769  0.21649769  0.21649769]
    #  [-0.55173486  0.27294404  0.27294404  0.27294404]
    #  [-0.40481846  0.31674737  0.31674737  0.31674737]
    #  [-0.24772241  0.34853185  0.34853185  0.34853185]
    #  [-0.08333027  0.36364422  0.36364422  0.36364422]
    #  [ 0.08339398  0.36451369  0.36451369  0.36451369]
    #  [ 0.24756571  0.34771203  0.34771203  0.34771203]
    #  [ 0.40505775  0.31747118  0.31747118  0.31747118]
    #  [ 0.55147519  0.27234468  0.27234468  0.27234468]
    #  [ 0.68533364  0.21694835  0.21694835  0.21694835]
    #  [ 0.80441694  0.15126347  0.15126347  0.15126347]
    #  [ 0.90933041  0.07870128  0.07870128  0.07870128]
    #  [ 1.          0.          0.          0.        ]]
    print(curve.get_points())


if __name__ == '__main__':
    main()
