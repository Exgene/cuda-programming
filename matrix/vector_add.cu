#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define THREAD_POOL 256
#define N 20000

void vector_add_cpu(double *a, double *b, double *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vector_add_gpu(double *a, double *b, double *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void init_vector(double *a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = (double)rand() / RAND_MAX;
  }
}

double get_time() {
  struct timespec st;
  clock_gettime(CLOCK_MONOTONIC, &st);
  return st.tv_sec + st.tv_nsec * 1e-9;
}

int main() {
  double *h_a, *h_b, *h_c_cpu, *h_c_cuda;
  double *d_a, *d_b, *d_c;
  size_t SIZE = sizeof(double) * N;

  h_a = (double *)malloc(SIZE);
  h_b = (double *)malloc(SIZE);
  h_c_cpu = (double *)malloc(SIZE);
  h_c_cuda = (double *)malloc(SIZE);

  init_vector(h_a, N);
  init_vector(h_b, N);

  srand(time(NULL));
  cudaMalloc(&d_a, SIZE);
  cudaMalloc(&d_b, SIZE);
  cudaMalloc(&d_c, SIZE);

  cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

  int numBlocks = (N - 1 + THREAD_POOL) / THREAD_POOL;

  std::cout << "Starting Warm Up Runs!" << "\n";

  for (int i = 0; i < 3; i++) {
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    vector_add_gpu<<<numBlocks, THREAD_POOL>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  std::cout << "Starting CPU Impl" << "\n";
  double cpu_time = 0.0;

  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    double end_time = get_time();
    cpu_time += end_time - start_time;
  }
  cpu_time = cpu_time / 20.0;
  std::cout << "It took" << cpu_time << "\n";

  std::cout << "Starting GPU Impl" << "\n";
  double gpu_time = 0.0;

  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vector_add_gpu<<<numBlocks, THREAD_POOL>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_time += end_time - start_time;
  }

  gpu_time = gpu_time / 20.0;
  std::cout << "It took" << gpu_time << "\n";

  cudaMemcpy(h_c_cuda, d_c, SIZE, cudaMemcpyDeviceToHost);

  bool is_wrong = false;
  for (int i = 0; i < N; i++) {
    if (fabs(h_c_cuda[i] - h_c_cpu[i]) > 1e-5) {
      printf("Value of cpu h_c_cpu: %f, Value of h_c_cuda: %f", h_c_cpu[i],
             h_c_cuda[i]);
      is_wrong = true;
      break;
    }
  }

  std::cout << "ISWRONG" << (is_wrong ? "WRONG" : "RIGHT") << "\n";
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_cuda);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
