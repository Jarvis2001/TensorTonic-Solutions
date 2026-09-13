#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int colID = blockIdx.x * blockDim.x + threadIdx.x;
    int rowID = blockIdx.y * blockDim.y + threadIdx.y;
    int elemId;

    if (rowID < M && colID < N) {
        elemId = rowID * N + colID;
        C[elemId] = A[elemId] + B[elemId];
    }
    
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
