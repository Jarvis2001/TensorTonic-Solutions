#include <cuda_runtime.h>
#include <math.h>

//A more optimized version for more kernel throughput
//__restrict__ qualifiers promise the compiler that input and output pointers don't overlap in memory, enabling better load/store.
__global__ void swish_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {

    //Calculates the global unique 1D thread ID across the entire grid.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //Calculates the total number of the threads launced across the grid. Used for grid-stride loops to allow threads to step through the dataset iteratively.
    int stride = blockDim.x * gridDim.x;

    //Processing 4 elemets at a time (128 bit loads/stores)
    //Calculates how many full 4-element vectors (float4) vector types, allowing the GPU to fetch and store 4 floats in a single memory transaction
    int vec_N = N  / 4;

    //Cast the standard 32-bit float pointers to 128bit float4 vector types, allowing the GPU to fetch and store 4 floats in a single memory transaction.
    const float4* in_vec = reinterpret_cast<const float4*>(input);    
    float4* out_vec = reinterpret_cast<float4*>(output);

    ///Grid-stride loop iterating over the float4 vector array chunks. Each thread processes vector index i and jumps stride.
    for (int i = idx; i < vec_N; i += stride) {

        //Loads 128-bit (4 consecutive float values) from global memory into register variables(val.x, val.y, val.z, val.w)
        float4 val = in_vec[i];

        //Computes the swish funtion for each of the 4 vector components. __expf is a hardware Special Funtion Unit (SFU) for more execution speed.
        val.x = val.x / (1.0f + __expf(-val.x));
        val.y = val.y / (1.0f + __expf(-val.y));
        val.z = val.z / (1.0f + __expf(-val.z));
        val.w = val.w / (1.0f + __expf(-val.w));

        //Writes all 4 updated values back to global memory in a single 128-bit store instruction.
        out_vec[i] = val;
    }

    //Handling remaining tail elements
    //Determines the starting index for the remaining elements if N is not evenly divisible by 4
    //Loops through any leftover elements that couldn't fit in the 128-bit float4 vector.
    int remainder_start = vec_N * 4;
    for (int i = remainder_start + idx; i < N; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + __expf(-x));
    }
    
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
