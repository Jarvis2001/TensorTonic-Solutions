#include <cuda_runtime.h>

#define TILE 32

__global__ void __launch_bounds__(256)
matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE + 1];  // +1 avoids bank conflicts
    __shared__ float Bs[TILE][TILE + 1];

    const int row0 = blockIdx.y * TILE;
    const int col0 = blockIdx.x * TILE;
    if (row0 >= M || col0 >= N) return;   // over-provisioned blocks exit cheaply

    const int tRow = threadIdx.y;
    const int tCol = threadIdx.x;
    const int tid  = tRow * 16 + tCol;
    const int r0 = tRow * 2, c0 = tCol * 2;   // thread's 2x2 outputs

    float acc[2][2] = {};

    for (int t = 0; t < K; t += TILE) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {         // 32x32 = 1024 elems / 256 threads
            int idx = i * 256 + tid;
            int r = idx >> 5, c = idx & 31;
            int ar = row0 + r, ac = t + c;
            As[r][c] = (ar < M && ac < K) ? A[(size_t)ar * K + ac] : 0.f;
            int br = t + r, bc = col0 + c;
            Bs[r][c] = (br < K && bc < N) ? B[(size_t)br * N + bc] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            float a0 = As[r0][kk],     a1 = As[r0 + 1][kk];
            float b0 = Bs[kk][c0],     b1 = Bs[kk][c0 + 1];
            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
        }
        __syncthreads();
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