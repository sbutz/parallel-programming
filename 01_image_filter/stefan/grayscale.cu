#include "jpeg.h"
#include "util.h"
#include <cstdint>
#include <cuda.h>
#include <iostream>

__global__ void RgbToGrayscale(unsigned char *inputImage, unsigned char *outputImage, int width,
                               int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        int rgb_idx = idx * 3;

        unsigned char r = inputImage[rgb_idx];
        unsigned char g = inputImage[rgb_idx + 1];
        unsigned char b = inputImage[rgb_idx + 2];

        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        outputImage[idx] = gray;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.jpg" << std::endl;
        return 1;
    }

    cudaInit();

    const char *inputFilename = argv[1];
    const char *outputFilename = argv[2];

    auto hInputImage = Jpeg::FromFile(inputFilename);
    auto height = hInputImage.GetHeight();
    auto width = hInputImage.GetWidth();
    auto channels = hInputImage.GetChannels();
    ASSERT(channels == 3, "Expecting an rgb image");

    unsigned char *dInputImage, *dOutputImage;
    CUDA_ASSERT(cudaMalloc((void **)&dInputImage, width * height * channels));
    CUDA_ASSERT(cudaMalloc((void **)&dOutputImage, width * height));

    CUDA_ASSERT(cudaMemcpy(dInputImage, hInputImage.GetRawData(), width * height * channels,
                           cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    RgbToGrayscale<<<gridSize, blockSize>>>(dInputImage, dOutputImage, width, height);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    Jpeg hOutputImage{width, height, 1};
    CUDA_ASSERT(cudaMemcpy(hOutputImage.GetRawData(), dOutputImage, width * height,
                           cudaMemcpyDeviceToHost));
    hOutputImage.Save(outputFilename);

    CUDA_ASSERT(cudaFree(dInputImage));
    CUDA_ASSERT(cudaFree(dOutputImage));

    return 0;
}
