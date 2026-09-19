#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //Column index j(0 to N -1)
    int idy = blockIdx.y * blockDim.y + threadIdx.y; //Row index i(0 to M-1)
    if (idy < M && idx < N) {
        B[idx * M + idy] = A[idy * N + idx];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
