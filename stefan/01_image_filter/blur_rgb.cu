#include "jpeg.h"
#include "util.h"
#include <cstdint>
#include <cuda.h>
#include <iostream>

__global__ void Blur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int startX = max(x - margin, 0);
        int endX = min(x + margin, width);
        int startY = max(y - margin, 0);
        int endY = min(y + margin, height);

        float r = 0;
        float g = 0;
        float b = 0;
        for (int i = startY; i < endY; i++)
        {
            for (int j = startX; j < endX; j++)
            {
                r += inputImage[(i * width + j) * channels + 0];
                g += inputImage[(i * width + j) * channels + 1];
                b += inputImage[(i * width + j) * channels + 2];
            }
        }
        float n = (endX - startX) * (endY - startY);
        outputImage[(y * width + x) * channels + 0] = r / n;
        outputImage[(y * width + x) * channels + 1] = g / n;
        outputImage[(y * width + x) * channels + 2] = b / n;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.jpg" << std::endl;
        return 1;
    }

    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];

    auto hInputImage = Jpeg::FromFile(inputFilename);
    auto height = hInputImage.GetHeight();
    auto width = hInputImage.GetWidth();
    auto channels = hInputImage.GetChannels();
    Assert(channels == 3, "Expecting rgb image");

    unsigned char *dInputImage, *dOutputImage;
    GpuAssert(cudaMalloc((void**)&dInputImage, width * height * channels * sizeof(float)));
    GpuAssert(cudaMalloc((void**)&dOutputImage, width * height * channels * sizeof(float)));

    GpuAssert(cudaMemcpy(dInputImage, hInputImage.GetRawData(), width * height * channels, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    int margin = 8;

    Blur<<<gridSize, blockSize>>>(dInputImage, dOutputImage, width, height, channels, margin);
    GpuAssert(cudaGetLastError());
    GpuAssert(cudaDeviceSynchronize());

    Jpeg hOutputImage{width, height, channels};
    GpuAssert(cudaMemcpy(hOutputImage.GetRawData(), dOutputImage, width * height * channels, cudaMemcpyDeviceToHost));
    hOutputImage.Save(outputFilename);

    GpuAssert(cudaFree(dInputImage));
    GpuAssert(cudaFree(dOutputImage));

    return 0;
}
