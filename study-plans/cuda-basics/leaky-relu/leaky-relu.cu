#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : (val * alpha);
    }
    
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}