#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    // Implement the kernel to compute matrix-vector dot product
    // Each thread computes one element of the result vector
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sm = 0.0f;

        for (int col = 0; col < cols; col++) {
            sm += matrix[row * cols + col] * vector[col];
        }

        result[row] = sm;
    }
}


std::vector<float> matrix_dot_vector(
    const std::vector<std::vector<float>>& matrix,
    const std::vector<float>& vec
) {
    if (matrix.empty()) {
        return {};
    }

    int rows = static_cast<int>(matrix.size());
    int cols = static_cast<int>(vec.size());

   
    for (const auto& r : matrix) {
        if (r.size() != vec.size()) {
            return {-1};
        }
    }

    std::vector<float> flat_matrix;
    flat_matrix.reserve(
        static_cast<size_t>(rows) * cols
    );

    for (const auto& r : matrix) {
        flat_matrix.insert(
            flat_matrix.end(), r.begin(), r.end()
        );
    }

    if (cols == 0) {
        return std::vector<float>(rows, 0.0f);
    }

    float* d_matrix = nullptr;
    float* d_vector = nullptr;
    float* d_result = nullptr;

    size_t matrixByte =
        static_cast<size_t>(rows) * cols * sizeof(float);
    size_t vectorByte =
        static_cast<size_t>(cols) * sizeof(float);
    size_t resultByte =
        static_cast<size_t>(rows) * sizeof(float);

    // 1. Allocate device memory.
    cudaMalloc(&d_matrix, matrixByte);
    cudaMalloc(&d_vector, vectorByte);
    cudaMalloc(&d_result, resultByte);

    // 2. Copy data to device.
    cudaMemcpy(
        d_matrix, flat_matrix.data(), matrixByte,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_vector, vec.data(), vectorByte,
        cudaMemcpyHostToDevice
    );

    // 3. Launch kernel.
    int threads_per_block = 256;
    int blocks = (rows + threads_per_block - 1)
                 / threads_per_block;

    matrix_vector_dot_kernel<<<blocks, threads_per_block>>>(
        d_matrix, d_vector, d_result, rows, cols
    );

    // 4. Copy result back.
    std::vector<float> result(rows);

    cudaMemcpy(
        result.data(), d_result, resultByte,
        cudaMemcpyDeviceToHost
    );

    // 5. Free memory.
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return result;
}
