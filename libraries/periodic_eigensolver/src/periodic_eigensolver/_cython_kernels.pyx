# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

"""BLAS-backed kernels for large complex Arnoldi vectors."""

from scipy.linalg cimport cython_blas as blas


cpdef double arnoldi_step(
    double complex[::1, :] basis,
    double complex[::1, :] hessenberg,
    double complex[::1] work,
    Py_ssize_t column,
    double breakdown_tolerance,
):
    cdef Py_ssize_t size = basis.shape[0]
    cdef Py_ssize_t index
    cdef int blas_size
    cdef int increment = 1
    cdef double beta
    cdef double complex coefficient
    cdef double complex correction
    cdef double complex update
    cdef double complex inverse_beta

    if size == 0:
        raise ValueError("Arnoldi vectors must not be empty.")
    if size > 2147483647:
        raise OverflowError("Arnoldi vector is too large for the BLAS integer ABI.")
    if work.shape[0] != size:
        raise ValueError("work must have the same length as a basis column.")
    if column < 0 or column + 1 >= basis.shape[1]:
        raise ValueError("column is outside the allocated Arnoldi basis.")
    if hessenberg.shape[0] <= column + 1 or hessenberg.shape[1] <= column:
        raise ValueError("hessenberg is too small for the requested column.")

    blas_size = <int>size
    with nogil:
        for index in range(column + 1):
            coefficient = blas.zdotc(
                &blas_size,
                &basis[0, index],
                &increment,
                &work[0],
                &increment,
            )
            hessenberg[index, column] = coefficient
            update = -coefficient
            blas.zaxpy(
                &blas_size,
                &update,
                &basis[0, index],
                &increment,
                &work[0],
                &increment,
            )

        for index in range(column + 1):
            correction = blas.zdotc(
                &blas_size,
                &basis[0, index],
                &increment,
                &work[0],
                &increment,
            )
            coefficient = hessenberg[index, column]
            hessenberg[index, column] = coefficient + correction
            update = -correction
            blas.zaxpy(
                &blas_size,
                &update,
                &basis[0, index],
                &increment,
                &work[0],
                &increment,
            )

        beta = blas.dznrm2(&blas_size, &work[0], &increment)
        hessenberg[column + 1, column] = beta
        if beta > breakdown_tolerance:
            blas.zcopy(
                &blas_size,
                &work[0],
                &increment,
                &basis[0, column + 1],
                &increment,
            )
            inverse_beta = 1.0 / beta
            blas.zscal(
                &blas_size,
                &inverse_beta,
                &basis[0, column + 1],
                &increment,
            )
    return beta


cpdef relative_residuals(
    double complex[:, ::1] ax,
    double complex[:, ::1] bx,
    double complex[::1] eigenvalues,
    double[::1] output,
    double complex[::1] work,
):
    cdef Py_ssize_t row_count = ax.shape[0]
    cdef Py_ssize_t mode_count = ax.shape[1]
    cdef Py_ssize_t index
    cdef int blas_rows
    cdef int source_increment
    cdef int work_increment = 1
    cdef int scalar_size = 1
    cdef double ax_norm
    cdef double bx_norm
    cdef double residual_norm
    cdef double eigenvalue_norm
    cdef double scale
    cdef double complex update

    if bx.shape[0] != row_count or bx.shape[1] != mode_count:
        raise ValueError("ax and bx must have identical shapes.")
    if eigenvalues.shape[0] != mode_count or output.shape[0] != mode_count:
        raise ValueError("eigenvalue and output lengths must match the mode count.")
    if work.shape[0] != row_count:
        raise ValueError("work must have one entry per matrix row.")
    if row_count == 0:
        raise ValueError("Residual vectors must not be empty.")
    if row_count > 2147483647 or mode_count > 2147483647:
        raise OverflowError("Residual batch is too large for the BLAS integer ABI.")

    blas_rows = <int>row_count
    source_increment = <int>mode_count
    with nogil:
        for index in range(mode_count):
            ax_norm = blas.dznrm2(
                &blas_rows, &ax[0, index], &source_increment
            )
            bx_norm = blas.dznrm2(
                &blas_rows, &bx[0, index], &source_increment
            )
            blas.zcopy(
                &blas_rows,
                &ax[0, index],
                &source_increment,
                &work[0],
                &work_increment,
            )
            update = -eigenvalues[index]
            blas.zaxpy(
                &blas_rows,
                &update,
                &bx[0, index],
                &source_increment,
                &work[0],
                &work_increment,
            )
            residual_norm = blas.dznrm2(
                &blas_rows, &work[0], &work_increment
            )
            eigenvalue_norm = blas.dznrm2(
                &scalar_size, &eigenvalues[index], &work_increment
            )
            scale = ax_norm + eigenvalue_norm * bx_norm
            if scale == 0.0:
                scale = 1.0
            output[index] = residual_norm / scale
