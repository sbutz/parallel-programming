#include "jpeg.h"
#include "util.h"
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <iostream>


__global__ void Blur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;

    if (x < width && y < height) {
        int startX = max(x - margin, 0);
        int endX = min(x + margin, width);
        int startY = max(y - margin, 0);
        int endY = min(y + margin, height);

        float v = 0;
        for (int i = startY; i < endY; i++)
        {
            for (int j = startX; j < endX; j++)
            {
                v += inputImage[(i * width + j) * channels + channel];
            }
        }
        float n = (endX - startX) * (endY - startY);
        outputImage[(y * width + x) * channels + channel] = v / n;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " margin input.jpg output.jpg" << std::endl;
        return 1;
    }

    int margin = std::atoi(argv[1]);
    const char* inputFilename = argv[2];
    const char* outputFilename = argv[3];

    auto hInputImage = Jpeg::FromFile(inputFilename);
    auto height = hInputImage.GetHeight();
    auto width = hInputImage.GetWidth();
    auto channels = hInputImage.GetChannels();

    unsigned char *dInputImage, *dOutputImage;
    CUDA_ASSERT(cudaMalloc((void**)&dInputImage, width * height * channels * sizeof(float)));
    CUDA_ASSERT(cudaMalloc((void**)&dOutputImage, width * height * channels * sizeof(float)));

    float time;
    cudaEvent_t start, stop;
    CUDA_ASSERT(cudaEventCreate(&start) );
    CUDA_ASSERT(cudaEventCreate(&stop) );
    CUDA_ASSERT(cudaEventRecord(start, 0) );

    CUDA_ASSERT(cudaMemcpy(dInputImage, hInputImage.GetRawData(), width * height * channels, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaEventRecord(stop, 0) );
    CUDA_ASSERT(cudaEventSynchronize(stop) );
    CUDA_ASSERT(cudaEventElapsedTime(&time, start, stop) );

    std::cout << "Elapsed time: " << time << "ms" << std::endl;


    dim3 blockSize(16, 16, channels);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    Blur<<<gridSize, blockSize>>>(dInputImage, dOutputImage, width, height, channels, margin);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    Jpeg hOutputImage{width, height, channels};
    CUDA_ASSERT(cudaMemcpy(hOutputImage.GetRawData(), dOutputImage, width * height * channels, cudaMemcpyDeviceToHost));
    hOutputImage.Save(outputFilename);

    CUDA_ASSERT(cudaFree(dInputImage));
    CUDA_ASSERT(cudaFree(dOutputImage));

    return 0;
}
