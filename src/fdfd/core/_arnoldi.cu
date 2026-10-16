// Complex FP64 two-pass Gram-Schmidt building blocks. One reduction block
// per active basis column; updates remain entirely in device memory.
#include <cuComplex.h>
extern "C" __global__ void project(
    const cuDoubleComplex* basis, const cuDoubleComplex* work,
    cuDoubleComplex* coefficients, int n, int columns) {
    __shared__ cuDoubleComplex sums[256];
    int column = blockIdx.x;
    cuDoubleComplex sum = make_cuDoubleComplex(0., 0.);
    for (int row = threadIdx.x; row < n; row += blockDim.x)
        sum = cuCadd(sum, cuCmul(cuConj(basis[column*n + row]), work[row]));
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sums[threadIdx.x] = cuCadd(sums[threadIdx.x], sums[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) coefficients[column] = sums[0];
}
extern "C" __global__ void subtract_projection(
    const cuDoubleComplex* basis, cuDoubleComplex* work,
    const cuDoubleComplex* coefficients, int n, int columns) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    cuDoubleComplex value = work[row];
    for (int column = 0; column < columns; ++column)
        value = cuCsub(value, cuCmul(basis[column*n + row], coefficients[column]));
    work[row] = value;
}
