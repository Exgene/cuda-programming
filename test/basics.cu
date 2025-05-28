#include <stdio.h>

__global__ void whoami(void){
    int block_id =
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // floor number in this building (rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep)

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv) {
  const int b_x = 2, b_y = 3, b_z = 4;
  const int t_x = 4, t_y = 4, t_z = 4;

  int blocks_per_grid = b_x * b_y * b_z;
  int threads_per_block = t_x * t_y * t_z;

  printf("Blocks per grid: %d", blocks_per_grid);
  printf("Threads per block: %d", threads_per_block);
  printf("Total number of threads: %d", blocks_per_grid * threads_per_block);

  dim3 blocksPerGrid(b_x, b_y, b_z);
  dim3 threadsPerBlock(t_x, t_y, t_z);

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}
