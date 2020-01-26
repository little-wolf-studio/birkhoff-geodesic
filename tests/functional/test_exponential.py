from geodesic.curve_shorten import compute_geodesic
from geodesic.geometry import Curve
import numpy as np


def compute_analytic_exponential_geodesic(xs, alpha, dimension):
    """
    Args:
        xs: array of shape :math:`(N,)`
        alpha: float
        dimension: int
    """
    coefficient = 1 / alpha
    return coefficient * np.log(np.cos(alpha * xs) / np.cos(alpha))


def test_single_threaded_exponential_geodesic():
    # Set dimension of the problem
    dimension = 2

    # Set parameters for computation
    number_of_global_nodes = 32
    number_of_local_nodes = 8
    maximum_average_node_movement = 1e-4

    # Create start and end point NumPy arrays
    start_point = np.zeros(dimension)
    start_point[0] = -1

    end_point = np.zeros(dimension)
    end_point[0] = 1

    # Create constant vector n
    alpha = 0.65
    n = alpha * np.ones(dimension)
    n[0] = 0

    # Define function to describe metric coefficient
    def metric_coefficient(x):
        return np.exp(-np.inner(n, x))

    def metric_coefficient_gradient(x):
        return -n * np.exp(-np.inner(n, x))

    # Create curve object for calculation
    curve = Curve(start_point, end_point, number_of_global_nodes)

    # Apply curve shortening procedure to minimise length
    compute_geodesic(
            curve,
            local_num_nodes=number_of_local_nodes,
            tol=maximum_average_node_movement,
            metric=metric_coefficient,
            grad_metric=metric_coefficient_gradient,
            processes=1
    )

    computed_geodesic = curve.get_points()
    xs = curve.get_points()[:, 0]
    analytical_geodesic = compute_analytic_exponential_geodesic(xs, alpha,
                                                                    dimension)

    for dim in range(1, dimension):
        mean_error = np.abs(analytical_geodesic - computed_geodesic[:, dim]).mean()
        print(mean_error)
        assert mean_error < 1e-4
