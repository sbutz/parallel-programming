#include <cuda.h>
#include <cuda_runtime.h>
#include "get_image.h"
#include "write_image.h"

__global__ void rgb_to_gray_kernel(float *input, float *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) return;

    int idx = row * width + col;
    output[idx] = 0.21f * input[3 * idx] + 0.71f * input[3 * idx + 1] + 0.07f * input[3 * idx + 2];
}

int main(int argc, char **argv){

    int w=0,h=0, channel=0;
    float *data=NULL;
    float *in_d = NULL;
    float *out_d = NULL;
    float *out_h =NULL;
    const char* file="minion.jpg";
    const char* file1="minion_copy.jpg";
    const char* file2="minion_bw.jpg";
    /*load image in the form described in moodle */

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    get_image(file, &data, &w , &h, &channel); 
    out_h = (float*) malloc(sizeof(float)*w*h);

    int in_size = sizeof(float) * 3 * w * h;
    int out_size = sizeof(float) * w * h;
    cudaEventRecord(start);
	cudaMalloc((void **) &in_d, in_size);
	cudaMemcpy(in_d, data, in_size, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &out_d, out_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    dim3 dimGrid((w - 1) / 16 + 1, (h - 1) / 16 + 1, 1);
    dim3 dimBlock(16, 16, 1);
    cudaEventRecord(start);
    rgb_to_gray_kernel<<<dimGrid, dimBlock>>> (in_d, out_d, w, h);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventRecord(start);
	cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    /* write black white image */
    write_JPEG_file_bw (file2, out_h, w, h);
    /*write color image */
  	write_JPEG_file_color(file1, data, w , h); 

    return 0;
}
