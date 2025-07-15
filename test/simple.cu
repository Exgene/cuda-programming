#include <cstdio>
__global__ void addArrays(int *a, int *b, int *c, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    printf("index:\n%d", index);
    c[index] = a[index] + b[index];
  }
}

int main() {
  int size = 1000;
  size_t bytes = size * sizeof(int);
  int *h_a = (int *)malloc(bytes);
  int *h_b = (int *)malloc(bytes);
  int *h_c = (int *)malloc(bytes);

  for (int i = 0; i < size; ++i) {
    h_a[i] = i;
    h_b[i] = 2 * i;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1, 1);

  addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
  cudaDeviceSynchronize();

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < size; ++i) {
  //   printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
  // }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
