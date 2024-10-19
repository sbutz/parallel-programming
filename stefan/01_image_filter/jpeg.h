#pragma once

#include "util.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <jpeglib.h>

class Jpeg
{
    using SizeType = std::uint32_t;
    using ValueType = JSAMPLE;
    using ImageType = std::vector<ValueType>;

  public:
    static Jpeg FromFile(const char* filename)
    {
        FILE* infile = fopen(filename, "rb");
        Assert(infile, "Failed to open file");

        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, infile);
        Assert(jpeg_read_header(&cinfo, TRUE) == 1, "Invalid Jpeg");
        jpeg_start_decompress(&cinfo);

        SizeType width = cinfo.output_width;
        SizeType height = cinfo.output_height;
        SizeType channels = cinfo.output_components;

        ImageType imageData;
        imageData.reserve(width * height * channels);

        while (cinfo.output_scanline < height) {
            JSAMPROW ptr = &imageData[cinfo.output_scanline * width * channels];
            jpeg_read_scanlines(&cinfo, &ptr, 1);
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);

        return Jpeg(std::move(imageData), width, height, channels);
    }

    Jpeg(SizeType width, SizeType height, SizeType channels)
        : data_{}, width_{width}, height_{height}, channels_{channels}
    {
        data_.reserve(width_*height_*channels_);
    }

    Jpeg(ImageType&& data, SizeType width, SizeType height, SizeType channels)
        :data_{std::move(data)}, width_{width}, height_{height}, channels_{channels}
    {
    }

    ValueType* GetRawData() { return data_.data(); }

    SizeType GetWidth() { return width_; }

    SizeType GetHeight() { return height_; }

    SizeType GetChannels() { return channels_; }

    void Save(const char* filename) {
        FILE* outfile = fopen(filename, "wb");
        Assert(outfile, "Failed to open file");

        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, outfile);

        cinfo.image_width = width_;
        cinfo.image_height = height_;
        Assert(channels_ == 1 || channels_ == 3, "Only rgb and grayscale supported");
        cinfo.input_components = channels_;
        cinfo.in_color_space = channels_ == 1 ? JCS_GRAYSCALE : JCS_RGB;

        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, 100, TRUE);
        jpeg_start_compress(&cinfo, TRUE);

        while (cinfo.next_scanline < height_) {
            JSAMPROW ptr = &data_[cinfo.next_scanline * width_*channels_];
            jpeg_write_scanlines(&cinfo, &ptr, 1);
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
    }

 private:
    std::vector<ValueType> data_;
    SizeType width_;
    SizeType height_;
    SizeType channels_;
};
