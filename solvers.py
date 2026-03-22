import inspect

import numpy as np


def print_array(array, array_name=None, end=None):
    """
    Nicely print a 2D NumPy array with dynamic precision formatting.

    This function prints a 2D array in a readable, formatted style. It
    automatically determines the required precision to accurately represent the
    array values, up to a maximum of 15 decimal places. The printed output
    includes the array name if provided; otherwise, the function attempts to
    infer the variable name from the calling scope.

    Parameters
    ----------
    array : numpy.ndarray
        A 2D NumPy array to print.
    array_name : str, optional
        The name to display for the array in the output. If ``None`` (default),
        the function attempts to infer the variable name from the caller's local
        variables. If it cannot be inferred, defaults to ``"array"``.
    end : str, optional
        String appended after the last value in each row, similar to the `end`
        parameter in Python's built-in ``print`` function. Default is ``None``,
        which adds a newline.
    """
    # find array_name
    if array_name is None:
        frame = inspect.currentframe().f_back
        for name, value in frame.f_locals.items():
            if value is array:
                array_name = name
                break
    if array_name is None:
        array_name = "array"

    # determine precision
    precision = 1
    while not np.allclose(array, np.round(array, precision)):
        precision = precision + 1
        if precision == 16:
            break
    format_str = f"{{:{precision + 3}.{precision}f}}"

    if len(array.shape) == 2:
        n, m = array.shape
    else:
        n = array.shape[0]

    for i in range(n):
        if i == 0:
            print(f"{array_name} = [ ", end="")
        else:
            print(" " * len(array_name) + "   [ ", end="")

        try:
            print(", ".join([format_str.format(v) for v in array[i]]), end=" ]")
        except TypeError:
            print(format_str.format(array[i]), end=" ]")
        print(end=end)


def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R


def gram_schmidt_eigen(A, maxiter=1000, verbose=False):
    """
    Compute the eigenvalues and eigenvectors of a square matrix using the QR
    algorithm with classical Gram-Schmidt QR factorisation.

    This function implements the basic QR algorithm:

    1. Factorise the matrix `A` into `Q` and `R` using Gram-Schmidt QR
       factorisation.
    2. Update the matrix as:

       .. math::
           A_{k+1} = R_k Q_k

    3. Accumulate the orthonormal transformations in `V` to compute the
       eigenvectors.
    4. Iterate until `A` becomes approximately upper triangular or until the
       maximum number of iterations is reached.

    Once the iteration converges, the diagonal of `A` contains the eigenvalues,
    and the columns of `V` contain the corresponding eigenvectors.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix. This matrix will be **modified in place** during the
        computation.
    maxiter : int, optional
        Maximum number of QR iterations to perform. Default is 100.
    verbose : bool, optional
        If ``True``, prints intermediate matrices (`A`, `Q`, `R`, and `V`) at
        each iteration. Useful for debugging and understanding convergence.
        Default is ``False``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        A 1D NumPy array of length ``n`` containing the eigenvalues of `A`.
        These are the diagonal elements of the final upper triangular matrix.
    V : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` whose columns are the normalized
        eigenvectors corresponding to the eigenvalues.
    it : int
        The number of iterations taken by the algorithm.
    """
    # identity matrix to store eigenvectors
    V = np.eye(A.shape[0])

    if verbose:
        print_array(A)

    it = -1
    for it in range(maxiter):
        if verbose:
            print(f"\n\n{it=}")

        # perform factorisation
        Q, R = gram_schmidt_qr(A)
        if verbose:
            print_array(Q)
            print_array(R)

        # update A and V in place
        A = R @ Q
        V = V @ Q

        if verbose:
            print_array(A)
            print_array(V)

        # test for convergence: is A upper triangular up to tolerance 1.0e-8?
        if np.allclose(A, np.triu(A), atol=1.0e-8):
            break

    eigenvalues = np.diag(A)
    return eigenvalues, V, it
