#include "jpeg.h"
#include "util.h"
#include <cstdint>
#include <cuda.h>
#include <iostream>

__global__ void BlurGrayscale(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int startX = max(x - margin, 0);
        int endX = min(x + margin, width - 1);
        int startY = max(y - margin, 0);
        int endY = min(y + margin, height - 1);

        float v = 0;
        for (int i = startY; i < endY; i++)
        {
            for (int j = startX; j < endX; j++)
            {
                v += inputImage[i * width + j];
            }
        }
        float n = (endX - startX) * (endY - startY);
        outputImage[y * width + x] = v / n;
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
    Assert(channels == 1, "Expecting Grayscale Image");

    unsigned char *dInputImage, *dOutputImage;
    GpuAssert(cudaMalloc((void**)&dInputImage, width * height * channels));
    GpuAssert(cudaMalloc((void**)&dOutputImage, width * height));

    GpuAssert(cudaMemcpy(dInputImage, hInputImage.GetRawData(), width * height * channels, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    int margin = 8;

    BlurGrayscale<<<gridSize, blockSize>>>(dInputImage, dOutputImage, width, height, margin);
    GpuAssert(cudaGetLastError());
    GpuAssert(cudaDeviceSynchronize());

    Jpeg hOutputImage{width, height, channels};
    GpuAssert(cudaMemcpy(hOutputImage.GetRawData(), dOutputImage, width * height, cudaMemcpyDeviceToHost));
    hOutputImage.Save(outputFilename);

    GpuAssert(cudaFree(dInputImage));
    GpuAssert(cudaFree(dOutputImage));

    return 0;
}
