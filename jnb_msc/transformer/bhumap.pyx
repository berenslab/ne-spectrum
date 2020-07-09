# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as np
import numpy as np
from cython.parallel import prange

cimport openTSNE.quad_tree
from openTSNE.quad_tree cimport QuadTree, Node, is_duplicate
from openTSNE._matrix_mul.matrix_mul cimport matrix_multiply_fft_1d, matrix_multiply_fft_2d

cdef double EPSILON = np.finfo(np.float64).eps


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log(double x) nogil
    double exp(double x) nogil
    double fabs(double x) nogil
    double fmax(double x, double y) nogil
    double isinf(long double) nogil
    double INFINITY


cpdef double estimate_negative_gradient_bh(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double eps=0.001,
    double theta=0.5,
    Py_ssize_t num_threads=1,
    bint pairwise_normalization=True,
):
    """Estimate the negative UMAP gradient using the Barnes Hut approximation.

    Notes
    -----
    Changes the gradient inplace to avoid needless memory allocation. As
    such, this must be run before estimating the positive gradients, since
    the negative gradient must be normalized at the end with the sum of
    q_{ij}s.

    """
    cdef:
        Py_ssize_t i, j, num_points = embedding.shape[0]
        double sum_Q = 0
        double[::1] sum_Qi = np.zeros(num_points, dtype=float)

    if num_threads < 1:
        num_threads = 1

    # In order to run gradient estimation in parallel, we need to pass each
    # worker it's own memory slot to write sum_Qs
    for i in prange(num_points, nogil=True, num_threads=num_threads, schedule="guided"):
        _estimate_negative_gradient_single(
            &tree.root, &embedding[i, 0], &gradient[i, 0], &sum_Qi[i], eps, theta
        )

    for i in range(num_points):
        sum_Q += sum_Qi[i]

    return sum_Q


cdef void _estimate_negative_gradient_single(
    Node * node,
    double * point,
    double * gradient,
    double * sum_Q,
    double eps,
    double theta,
) nogil:
    # Make sure that we spend no time on empty nodes or self-interactions
    if node.num_points == 0 or node.is_leaf and is_duplicate(node, point):
        return

    cdef:
        double distance = EPSILON
        double q_ij, tmp
        Py_ssize_t d

    # Compute the squared euclidean distance in the embedding space from the
    # new point to the center of mass
    for d in range(node.n_dims):
        tmp = node.center_of_mass[d] - point[d]
        distance += (tmp * tmp)

    # Check whether we can use this node as a summary
    if node.is_leaf or node.length / sqrt(distance) < theta:
        q_ij = 1 / (1 + distance)

        sum_Q[0] += node.num_points * q_ij

        q_ij = (1 / (eps + distance)) * q_ij

        for d in range(node.n_dims):
            gradient[d] -= node.num_points * q_ij * (point[d] - node.center_of_mass[d])

        return

    # Otherwise we have to look for summaries in the children
    for d in range(1 << node.n_dims):
        _estimate_negative_gradient_single(&node.children[d], point, gradient, sum_Q, eps, theta)


cpdef double estimate_negative_gradient_elastic(
    QuadTree tree,
    double[:, ::1] embedding,
    double[:, ::1] gradient,
    double theta=0.5,
    double dof=1,
    double eps=0.001,
    double elastic_const=10000,
    Py_ssize_t num_threads=1,
    bint pairwise_normalization=True,
):
    """Estimate the negative tSNE gradient using the Barnes Hut approximation.
    By setting `eps=0.001` and `elastic_const=1` the UMAP gradient
    (with no negative sampling) can be calculated.

    Notes
    -----
    Changes the gradient inplace to avoid needless memory allocation. As
    such, this must be run before estimating the positive gradients, since
    the negative gradient must be normalized at the end with the sum of
    q_{ij}s.

    """
    cdef:
        Py_ssize_t i, j, num_points = embedding.shape[0]
        double sum_Q = 0
        double[::1] sum_Qi = np.zeros(num_points, dtype=float)

    if num_threads < 1:
        num_threads = 1

    # In order to run gradient estimation in parallel, we need to pass each
    # worker it's own memory slot to write sum_Qs
    for i in prange(num_points, nogil=True, num_threads=num_threads, schedule="guided"):
        _estimate_negative_gradient_single(
            &tree.root, &embedding[i, 0], &gradient[i, 0], &sum_Qi[i], eps, theta
        )

    for i in range(num_points):
        sum_Q += sum_Qi[i]

    # Normalize q_{ij}s
    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if pairwise_normalization:
                gradient[i, j] /= elastic_const
            else:
                gradient[i, j] /= elastic_const

    return sum_Q
