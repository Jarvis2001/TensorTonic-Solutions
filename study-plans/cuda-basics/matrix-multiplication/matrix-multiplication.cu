#include <cuda_runtime.h>

#define TILE 32

__global__ void __launch_bounds__(256)
matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[2][TILE][TILE + 1];
    __shared__ float Bs[2][TILE][TILE + 1];

    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;
    if (row0 >= M || col0 >= N) return;

    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int r0  = threadIdx.y * 2, c0 = threadIdx.x * 2;

    float acc[2][2] = {};

    // prefetch first K-tile
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = i * 256 + tid;
        int r = idx >> 5, c = idx & 31;
        As[0][r][c] = (row0 + r < M && c < K)  ? __ldg(&A[(size_t)(row0 + r) * K + c])      : 0.f;
        Bs[0][r][c] = (r < K && col0 + c < N)  ? __ldg(&B[(size_t)r * N + col0 + c])       : 0.f;
    }
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += TILE) {
        if (t + TILE < K) {   // prefetch next tile into the other buffer
            int nt = t + TILE;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int idx = i * 256 + tid;
                int r = idx >> 5, c = idx & 31;
                As[cur^1][r][c] = (row0 + r < M && nt + c < K) ? __ldg(&A[(size_t)(row0 + r) * K + nt + c]) : 0.f;
                Bs[cur^1][r][c] = (nt + r < K && col0 + c < N) ? __ldg(&B[(size_t)(nt + r) * N + col0 + c]) : 0.f;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            float a0 = As[cur][r0][kk], a1 = As[cur][r0 + 1][kk];
            float b0 = Bs[cur][kk][c0], b1 = Bs[cur][kk][c0 + 1];
            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
        }
        __syncthreads();
        cur ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int row = row0 + r0 + i;
        if (row >= M) continue;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int col = col0 + c0 + j;
            if (col < N) C[(size_t)row * N + col] = acc[i][j];
        }
    }
}

// ---- solve unchanged ----
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}