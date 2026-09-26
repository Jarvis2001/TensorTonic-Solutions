#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row_base = blockIdx.y * BM + ty * TM;
    const int col_base = blockIdx.x * BN + tx * TN;

    float acc[TM][TN] = {0.0f};

    for (int kb = 0; kb < K; kb += BK) {

        // -------------------------------------------------
        // Load A tile: 64 x 16 = 1024 floats
        // 256 threads -> 4 values/thread
        // -------------------------------------------------
        int tid = ty * blockDim.x + tx;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear = tid + i * (blockDim.x * blockDim.y);

            int ar = linear / BK;
            int ak = linear % BK;

            int gr = blockIdx.y * BM + ar;
            int gk = kb + ak;

            As[ar][ak] =
                (gr < M && gk < K)
                ? A[gr * K + gk]
                : 0.0f;
        }

        // -------------------------------------------------
        // Load B tile: 16 x 64 = 1024 floats
        // -------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear = tid + i * (blockDim.x * blockDim.y);

            int bk = linear / BN;
            int bc = linear % BN;

            int gk = kb + bk;
            int gc = blockIdx.x * BN + bc;

            Bs[bk][bc] =
                (gk < K && gc < N)
                ? B[gk * N + gc]
                : 0.0f;
        }

        __syncthreads();

        // -------------------------------------------------
        // Compute 64x64 tile
        // -------------------------------------------------
        #pragma unroll
        for (int k = 0; k < BK; ++k) {

            float a[TM];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[ty * TM + i][k];

            float b[TN];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    // -------------------------------------------------
    // Store
    // -------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            int r = row_base + i;
            int c = col_base + j;

            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
