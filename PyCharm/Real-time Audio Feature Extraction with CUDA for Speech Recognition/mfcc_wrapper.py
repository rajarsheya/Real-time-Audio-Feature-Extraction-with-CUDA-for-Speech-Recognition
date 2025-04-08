import ctypes
import numpy as np
import os

# Load the compiled CUDA DLL (change the path to where you placed the DLL)
#dll_path = os.path.join(os.path.dirname(__file__), '..', 'visual_studio', 'x64', 'Release', 'libmfcc.dll')
#mfcc = ctypes.CDLL(dll_path)
dll_path = "D:\\CUDA Projects\\Real-time Audio Feature Extraction with CUDA for Speech Recognition\\Visual Studio\\mfcc_cuda\\x64\\Release\\libmfcc.dll"
mfcc = ctypes.CDLL(dll_path)


def compute_mfcc_cuda(frame: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = np.ascontiguousarray(frame, dtype=np.float32)  # Make sure the frame is in contiguous memory format
    output = np.zeros(13, dtype=np.float32)  # 13 MFCCs

    # Define argument types for the CUDA function
    mfcc.compute_mfcc.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    ]

    # Define return type (None since it's a void function)
    mfcc.compute_mfcc.restype = None

    # Call the CUDA function
    mfcc.compute_mfcc(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(frame),
        sample_rate,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )

    return output
