#include "jpeg.h"
#include "util.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

static constexpr std::size_t N_ITERATIONS = 100;

__global__ void Blur(unsigned char *inputImage, unsigned char *outputImage, int width, int height,
                     int channels, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;

    if (x < width && y < height) {
        int startX = max(x - margin, 0);
        int endX = min(x + margin, width);
        int startY = max(y - margin, 0);
        int endY = min(y + margin, height);

        float v = 0;
        for (int i = startY; i < endY; i++) {
            for (int j = startX; j < endX; j++) {
                v += inputImage[(i * width + j) * channels + channel];
            }
        }
        float n = (endX - startX) * (endY - startY);
        outputImage[(y * width + x) * channels + channel] = v / n;
    }
}

void Filter(const char *inputFilename, const char *outputFilename, std::size_t margin) {
    auto hInputImage = Jpeg::FromFile(inputFilename);
    auto height = hInputImage.GetHeight();
    auto width = hInputImage.GetWidth();
    auto channels = hInputImage.GetChannels();

    unsigned char *dInputImage, *dOutputImage;
    CUDA_ASSERT(cudaMalloc((void **)&dInputImage, width * height * channels * sizeof(float)));
    CUDA_ASSERT(cudaMalloc((void **)&dOutputImage, width * height * channels * sizeof(float)));
    CUDA_ASSERT(cudaMemcpy(dInputImage, hInputImage.GetRawData(), width * height * channels,
                           cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16, channels);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    Blur<<<gridSize, blockSize>>>(dInputImage, dOutputImage, width, height, channels, margin);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    Jpeg hOutputImage{width, height, channels};
    CUDA_ASSERT(cudaMemcpy(hOutputImage.GetRawData(), dOutputImage, width * height * channels,
                           cudaMemcpyDeviceToHost));
    hOutputImage.Save(outputFilename);

    CUDA_ASSERT(cudaFree(dInputImage));
    CUDA_ASSERT(cudaFree(dOutputImage));
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " margin input.jpg output.jpg" << std::endl;
        return 1;
    }

    cudaInit();

    int margin = std::atoi(argv[1]);
    const char *inputFilename = argv[2];
    const char *outputFilename = argv[3];

    std::size_t input_size = std::atoi(argv[1]);
    for (auto i = 0; i < N_ITERATIONS; i++) {
        Filter(inputFilename, outputFilename, margin);
    }

    return 0;
}
