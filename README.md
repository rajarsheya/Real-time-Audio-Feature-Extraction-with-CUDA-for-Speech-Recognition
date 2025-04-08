
# Real-time Audio Feature Extraction with CUDA for Speech Recognition

## Project Overview
This project accelerates the extraction of audio features like Mel-frequency cepstral coefficients (MFCCs) using CUDA. The goal is to perform real-time feature extraction from live audio streams, enabling speech-to-text systems to process audio in real-time. This implementation leverages the parallel processing power of CUDA to significantly improve performance compared to traditional CPU-based methods.

## Demo Link: https://youtu.be/xioLZvnm0xY

### Key Features:
- Real-time audio capture from a microphone.
- CUDA-accelerated extraction of MFCCs from audio data.
- Efficient pipeline for real-time speech recognition.
  
## CUDA Implementation:
The CUDA-based MFCC feature extraction is implemented using a C++ code, which is compiled into a shared library (`libmfcc.dll`):

### Explanation:
1. **CUDA Kernel - apply_hamming**: This kernel applies a Hamming window to the audio signal. The Hamming window is used to reduce spectral leakage during the FFT computation.
2. **CUDA Kernel - dummy_fft**: This kernel simulates the FFT computation by applying a logarithmic transformation. A full FFT implementation would be necessary in a complete pipeline.
3. **compute_mfcc function**: This function is the main CUDA function that allocates memory on the GPU, applies the Hamming window, performs a simulated FFT, and returns the first 13 MFCCs.

The C++ code is compiled and linked into a shared library (`libmfcc.dll`) that can be used in the Python code through the `ctypes` library.

## Project Setup in Visual Studio:
1. **CUDA Toolkit Installation**: Ensure that you have the appropriate version of the CUDA Toolkit installed on your system. This project uses CUDA 11.x.
2. **Setting Up the CUDA Project**: In Visual Studio:
   - Create a new CUDA project.
   - Add the `compute_mfcc` function and its dependencies to the project.
   - Set the target architecture to match your GPU.
3. **Build the DLL**: After setting up the project, build it to generate the `libmfcc.dll` file, which will be used in the Python script.

## Files:
- **capture_audio.py**: Captures audio from the microphone and processes it using CUDA to extract MFCC features.
- **mfcc_wrapper.py**: A wrapper for the CUDA-based MFCC feature extraction.
- **visualize_mfcc.py**: Visualizes the extracted MFCCs over time (optional, can be removed if only processing is needed).
- **test_dll_loading.py**: A utility to test loading the DLL for the CUDA-based MFCC extraction.

## Prerequisites:
- Python 3.x
- CUDA Toolkit 11.x or higher
- `sounddevice` for audio capture
- `numpy`, `matplotlib` (for visualization, optional)
- `ctypes` for interacting with the CUDA DLL
- `PyCharm` or Visual Studio for running the project (with CUDA support enabled)

## How to Run the Code:

### 1. Setup and Dependencies:
First, ensure the following dependencies are installed:

```bash
pip install sounddevice numpy matplotlib
```

For CUDA:
- Make sure that you have the appropriate CUDA Toolkit installed and configured.
- The `mfcc_wrapper.py` will load the `libmfcc.dll` file, which should be generated from the C++ CUDA code.

### 2. Running the Capture Script:
The **capture_audio.py** script captures audio from your microphone and processes it in real-time. To run this script, execute the following command:

```bash
python capture_audio.py
```

This will start the audio capture and begin processing MFCC features, printing them in the terminal.

### 3. (Optional) Visualizing the MFCCs:
To visualize the extracted MFCCs, you can run the **visualize_mfcc.py** script. This will display a graph of the MFCCs in real-time:

```bash
python visualize_mfcc.py
```

This will show a real-time plot of the MFCCs being generated from the live audio stream.

### 4. Testing the DLL Loading:
You can also test whether the DLL is loading correctly using the **test_dll_loading.py** script:

```bash
python test_dll_loading.py
```

This will check if the CUDA-accelerated MFCC feature extraction library is properly loaded and working.

## Troubleshooting:
- **DLL not found**: Ensure that the path to the `libmfcc.dll` file is correctly specified in the `mfcc_wrapper.py` script.
- **CUDA errors**: Ensure that your system has a compatible GPU and CUDA drivers installed.

## Authors:
- Arsheya Raj
