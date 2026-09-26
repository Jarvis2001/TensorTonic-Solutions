#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Each thread computes 2x2 outputs
    const int row0 = blockIdx.y * TILE_M + 2 * ty;
    const int row1 = row0 + 1;

    const int col0 = blockIdx.x * TILE_N + 2 * tx;
    const int col1 = col0 + 1;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile) {

        const int k_base = tile * TILE_K;

        // ------------------------------------------------------------
        // Load A: 32x16 tile
        // 256 threads load 512 elements => 2 per thread
        // ------------------------------------------------------------
        int a_k = k_base + tx;

        if (row0 < M && a_k < K)
            As[2 * ty][tx] = A[row0 * K + a_k];
        else
            As[2 * ty][tx] = 0.0f;

        if (row1 < M && a_k < K)
            As[2 * ty + 1][tx] = A[row1 * K + a_k];
        else
            As[2 * ty + 1][tx] = 0.0f;

        // ------------------------------------------------------------
        // Load B: 16x32 tile
        // 256 threads load 512 elements => 2 per thread
        // ------------------------------------------------------------
        int b_row = k_base + ty;

        if (b_row < K && col0 < N)
            Bs[ty][2 * tx] = B[b_row * N + col0];
        else
            Bs[ty][2 * tx] = 0.0f;

        if (b_row < K && col1 < N)
            Bs[ty][2 * tx + 1] = B[b_row * N + col1];
        else
            Bs[ty][2 * tx + 1] = 0.0f;

        __syncthreads();

        // ------------------------------------------------------------
        // Compute 2x2 output tile
        // ------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a0 = As[2 * ty][k];
            float a1 = As[2 * ty + 1][k];

            float b0 = Bs[k][2 * tx];
            float b1 = Bs[k][2 * tx + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // ------------------------------------------------------------
    // Store
    // ------------------------------------------------------------
    if (row0 < M && col0 < N)
        C[row0 * N + col0] = c00;

    if (row0 < M && col1 < N)
        C[row0 * N + col1] = c01;

    if (row1 < M && col0 < N)
        C[row1 * N + col0] = c10;

    if (row1 < M && col1 < N)
        C[row1 * N + col1] = c11;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
