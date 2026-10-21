#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    // Implement the kernel for matrix multiplication C = A * B
    // A is M x K, B is K x N, C is M x N
    
    //thread
    int idx  = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= M*N) return;

    int row = idx/N;
    int col = idx%N;

    float sm = 0.0f;
    
    for(int k = 0; k < K ;k++){
        sm += A[row*K + k] * B[k*N + col];

    }

    C[row*N + col] = sm;
}

std::vector<std::vector<float>> matrixmul(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b
) {
    // Return {-1} if dimensions don't align

    int row_a = a[0].size();
    int row_b = b.size();
    if(row_a != row_b)return {{-1}};

    
    int M = static_cast<int>(a.size());
    int K = static_cast<int>(a[0].size());
    int N = static_cast<int>(b[0].size());

    vector<float> h_A;
    vector<float> h_B;
    h_A.reserve(M * K);
    h_B.reserve(K * N);

    for (const auto& row : a)
        h_A.insert(h_A.end(), row.begin(), row.end());

    for (const auto& row : b)
        h_B.insert(h_B.end(), row.begin(), row.end());
    
    vector<float> h_C(M * N, 0.0f);

    float *a_ptr = nullptr;
    float *b_ptr = nullptr;
    float *c_ptr = nullptr;

    // 1. Allocate device memory

    
    auto a_byte_size = a.size()*a[0].size()*sizeof(float);
    auto b_byte_size = b.size()*b[0].size()*sizeof(float);
    auto c_byte_size = M * N * sizeof(float);

    cudaMalloc(&a_ptr, a_byte_size);
    cudaMalloc(&b_ptr, b_byte_size);
    cudaMalloc(&c_ptr, c_byte_size);


    // 2. Copy data to device
    cudaMemcpy(a_ptr, h_A.data(), a_byte_size,cudaMemcpyHostToDevice);

    cudaMemcpy(b_ptr, h_B.data(), b_byte_size,cudaMemcpyHostToDevice);
    
    // 3. Launch kernel
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    matmul_kernel<<<blocks, threads>>>(a_ptr, b_ptr, c_ptr, M, K, N);

    // 4. Copy result back
    cudaMemcpy(h_C.data(), c_ptr, c_byte_size ,cudaMemcpyDeviceToHost);


    // 5. Free memory and return result
    cudaFree(a_ptr);
    a_ptr = nullptr;
    cudaFree(b_ptr);
    b_ptr = nullptr;
    cudaFree(c_ptr);
    c_ptr = nullptr;

    vector<vector<float>> C(M, vector<float>(N));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = h_C[i * N + j];
        }
    } 

    return C;
}