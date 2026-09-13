#include <cuda_runtime.h>
#include <cstdint>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    //A more optimized version
    int total_elements = M * N;
    int total_vec = total_elements / 4;
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
    int stride = (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y);
    
    if (total_elements % 4 == 0 && 
        reinterpret_cast<uintptr_t>(A) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(B) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(C) % 16 == 0) {

        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4* C4 = reinterpret_cast<float4*>(C);        

        for (int i = tid; i < total_vec; i += stride) {
            float4 a = A4[i];
            float4 b = B4[i];
            float4 c;
            c.x = a.x + b.x;
            c.y = a.y + b.y;
            c.z = a.z + b.z;
            c.w = a.w + b.w;
            C4[i] = c;
        }
    } else {
        for (int i = tid; i < total_elements; i += stride) {
            C[i] = A[i] + B[i];
        }
    }
        
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
