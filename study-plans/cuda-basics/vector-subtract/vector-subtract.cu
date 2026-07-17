#include <cuda_runtime.h>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {
    //PTX:
    //mov.u32 %ctaid.x
    //mov.u32 %ntid.x
    //mov.u32 %tid.x

    //Computing global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //setp.ge.s32 %p1, %r1, %r2
    //@%p1 bra ...

    //Exit threads that fall outside the vector length
    if (idx >= N) {
        return;
    }

    //mul.wide.s32 idx, 4
    //ld.global.f43 A[idx]
    //ld.global.f32 B[idx]
    //sub.f32
    //st.global.f32 C[idx]

    //Load on float from A and B, subtract them, and store
    //the result to the corresponding position in C.
    C[idx] = A[idx] - B[idx];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_sub<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}