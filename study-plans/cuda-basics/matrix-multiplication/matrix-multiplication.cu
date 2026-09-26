#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int colID = blockIdx.x * blockDim.x + threadIdx.x;
    int rowID = blockIdx.y * blockDim.y + threadIdx.y;

    if (rowID < M && colID < N) {
        float sum = 0.0f;        
        for (int k = 0; k < K; ++k) {
            sum += A[rowID * K + k] * B[k * N + colID];
        }
        C[rowID * N + colID] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
