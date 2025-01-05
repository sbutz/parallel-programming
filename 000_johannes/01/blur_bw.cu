#include <cuda.h>
#include <cuda_runtime.h>
#include "get_image.h"
#include "write_image.h"

__global__ void bw_blur_kernel(
  float * input, float *output, int width, int height, int margin
) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height || col >= width) return;

  int const lowerRow = row > margin ? row - margin : 0;
  int const upperRow = height - row > margin ? row + margin : height - 1;
  int const lowerCol = col > margin ? col - margin : 0;
  int const upperCol = width - col > margin ? col + margin : width - 1;
  float sum = 0.0;
  for (int i = lowerRow; i <= upperRow; ++i) {
    for (int j = lowerCol; j <= upperCol; ++j) {
      sum += input[i * width + j];
    }
  }
  float n = (float) (upperRow - lowerRow + 1) * (float) (upperCol - lowerCol + 1);
  output [row * width + col] = sum / n;
}


int main(int argc, char ** argv){
  int w=0, h=0, channel=0;
  float * data = NULL;
  float * in_d = NULL;
  float * out_d = NULL;
  float * out_h = NULL;
  const char * file_in = "minion_bw.jpg";
  const char * file_out = "minion_bw_blurred2.jpg";

  int const margin = 10;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  get_image(file_in, &data, &w , &h, &channel);

  int const in_size = sizeof(float) * w * h;
  int const out_size = sizeof(float) * w * h;

  out_h = (float *) malloc(out_size);

  cudaEventRecord(start);
	cudaMalloc((void **) &in_d, in_size);
	cudaMemcpy(in_d, data, in_size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &out_d, out_size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  dim3 dimGrid((w - 1) / 16 + 1, (h - 1) / 16 + 1, 1);
  dim3 dimBlock(16, 16, 1);
  cudaEventRecord(start);
  bw_blur_kernel<<<dimGrid, dimBlock>>> (in_d, out_d, w, h, margin);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaEventRecord(start);
	cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
  cudaFree(in_d);
  cudaFree(out_d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  write_JPEG_file_bw (file_out, out_h, w, h);

  return 0;
}
