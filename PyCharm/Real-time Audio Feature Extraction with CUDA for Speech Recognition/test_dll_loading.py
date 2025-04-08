import ctypes

# Absolute path to libmfcc.dll
dll_path = "D:\\CUDA Projects\\Real-time Audio Feature Extraction with CUDA for Speech Recognition\\Visual Studio\\mfcc_cuda\\x64\\Release\\libmfcc.dll"

try:
    mfcc = ctypes.CDLL(dll_path)
    print("DLL loaded successfully")
except OSError as e:
    print(f"Error loading DLL: {e}")
