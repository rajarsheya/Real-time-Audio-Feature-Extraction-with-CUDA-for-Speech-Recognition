//--------------------------------------------------------------------------------------------------------------------------------------------------
// Project : Real-time-Audio-Feature-Extraction-with-CUDA-for-Speech-Recognition
// Implement a system that uses CUDA to accelerate the extraction of audio features (e.g., MFCCs) from live audio streams.
// Author: Arsheya Raj
// Date: 8th April 2025
//--------------------------------------------------------------------------------------------------------------------------------------------------
//
//  Develop a system that utilizes CUDA to accelerate the extraction of audio features like Mel-frequency cepstral coefficients (MFCCs)
//  from live audio streams. This enables real-time speech recognition by providing pre-processed audio data to a speech-to-text model
//  running on local hardware.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#include <stdio.h>

extern "C" {
    __global__ void apply_hamming(float* signal, int len) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < len) {
            signal[i] *= 0.54f - 0.46f * cosf(2.0f * 3.14159265f * i / (len - 1));
        }
    }

    __global__ void dummy_fft(float* signal, int len) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < len) {
            signal[i] = logf(1 + fabsf(signal[i]));
        }
    }

    __declspec(dllexport)
        void compute_mfcc(float* signal, int len, int sample_rate, float* mfcc_out) {
        float* d_signal;
        cudaMalloc((void**)&d_signal, len * sizeof(float));
        cudaMemcpy(d_signal, signal, len * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (len + blockSize - 1) / blockSize;
        apply_hamming << <numBlocks, blockSize >> > (d_signal, len);
        dummy_fft << <numBlocks, blockSize >> > (d_signal, len);

        float temp[13];
        cudaMemcpy(temp, d_signal, 13 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 13; i++)
            mfcc_out[i] = temp[i];

        cudaFree(d_signal);
    }
}