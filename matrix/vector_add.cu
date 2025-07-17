#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10 // Vector size = 10 million
#define BLOCK_SIZE 256

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
    printf("C{%d}: %f \n", i, c[i]);
  }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
    // printf("C{%d}: %f \n", i, c[i]);
  }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
  for (int i = 0; i < n; i++) {
    vec[i] = (float)rand() / RAND_MAX;
  }
}

// Function to measure execution time
double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;
  size_t size = N * sizeof(float);

  srand(time(NULL));
  // Allocate host memory
  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c_cpu = (float *)malloc(size);
  h_c_gpu = (float *)malloc(size);

  // Initialize vectors
  init_vector(h_a, N);
  init_vector(h_b, N);

  // Allocate device memory
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy data to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // This is the "test" block you wrote
  float *test1, *test2;
  test1 = (float *)malloc(size);
  test2 = (float *)malloc(size);
  cudaMemcpy(test1, d_a, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(test2, d_b, size, cudaMemcpyDeviceToHost);

  // --- ADD THIS CORRECT VERIFICATION BLOCK ---
  printf("\n--- Verifying the cudaMemcpy operation for vector 'a' ---\n");
  bool copy_is_correct = true;
  for (int i = 0; i < N; i++) {
    // Directly compare the original value with the value after a round-trip
    if (h_a[i] != test1[i]) {
      printf("Mismatch at index %d! Original: %f, After Copy: %f\n", i, h_a[i],
             test1[i]);
      copy_is_correct = false;
    }
  }

  if (copy_is_correct) {
    printf("SUCCESS: The data copy for 'a' to/from the GPU is correct.\n");
  } else {
    printf("FAILURE: The data copy for 'a' is incorrect.\n");
    exit(0);
  }

  // Define grid and block dimensions
  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
  // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256
  // = 4 rounded

  // Warm-up runs
  // printf("Performing warm-up runs...\n");
  // for (int i = 0; i < 3; i++) {
  //   vector_add_cpu(h_a, h_b, h_c_cpu, N);
  //   vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
  //   cudaDeviceSynchronize();
  // }

  const int LOOP = 1;
  // Benchmark CPU implementation
  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < LOOP; i++) {
    double start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / LOOP * 1.0;

  // Benchmark GPU implementation
  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < LOOP; i++) {
    double start_time = get_time();
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / LOOP * 1.0;

  // Print results
  printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
  printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  // Verify results (optional)
  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
      // printf("GPU: %.2f: CPU: %.2f", h_c_gpu[i], h_c_cpu[i]);
      correct = false;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  // Free memory
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
