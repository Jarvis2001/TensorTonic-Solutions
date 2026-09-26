#include <cuda_runtime.h>

#define TILE 64   // block computes a 64x64 output tile
// threads stay 16x16 (fixed by solve), each computes 4x4 outputs

__global__ void __launch_bounds__(256)
matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tRow = threadIdx.y;      // 0..15
    const int tCol = threadIdx.x;      // 0..15
    const int tid  = tRow * 16 + tCol; // 0..255

    // grid is sized for 16x16 tiles, we do 64x64 -> stride over output tiles
    for (int row0 = blockIdx.y * TILE; row0 < M; row0 += gridDim.y * TILE) {
        for (int col0 = blockIdx.x * TILE; col0 < N; col0 += gridDim.x * TILE) {

            float acc[4][4] = {};

            for (int t = 0; t < K; t += TILE) {
                // cooperative load of 64x64 A-tile and B-tile (16 elements/thread)
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    int idx = i * 256 + tid;
                    int r = idx >> 6, c = idx & 63;
                    int ar = row0 + r, ac = t + c;
                    As[r][c] = (ar < M && ac < K) ? A[(size_t)ar * K + ac] : 0.f;
                    int br = t + r, bc = col0 + c;
                    Bs[r][c] = (br < K && bc < N) ? B[(size_t)br * N + bc] : 0.f;
                }
                __syncthreads();

                #pragma unroll
                for (int kk = 0; kk < TILE; ++kk) {
                    float a[4], b[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) a[i] = As[tRow * 4 + i][kk];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tCol * 4 + j];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
                __syncthreads();
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = row0 + tRow * 4 + i;
                if (row >= M) continue;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = col0 + tCol * 4 + j;
                    if (col < N) C[(size_t)row * N + col] = acc[i][j];
                }
            }
        }
    }
}

// ---- unchanged boilerplate ----
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}