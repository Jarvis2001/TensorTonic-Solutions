#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    //Well the previous one betrayed me
    int total_elements = M * N;

    if (total_elements % 4 == 0) {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);
    
        int total_vec = total_elements / 4;
        int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
    
        //Direct mapping without loop overhead
        if (tid  < total_vec) {
            float4 a = A4[tid];
            float4 b = B4[tid];
            C4[tid] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }        
    } else {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < M && col < N) {
            int idx = row * N + col;
            C[idx] = A[idx] + B[idx];            
        }
    }
    
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
