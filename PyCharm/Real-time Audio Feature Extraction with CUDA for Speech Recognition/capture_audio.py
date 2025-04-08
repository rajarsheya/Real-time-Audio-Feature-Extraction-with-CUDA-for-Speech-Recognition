import sounddevice as sd
import numpy as np
import soundfile as sf
from mfcc_wrapper import compute_mfcc_cuda

# Parameters
SAMPLE_RATE = 16000  # Standard sample rate for audio
FRAME_SIZE = 1024  # Number of samples per frame
HOP_SIZE = 512  # Overlap between consecutive frames

# File path to save MFCCs
OUTPUT_FILE = "mfcc_features.txt"


# Callback function that is called whenever new audio is available
def audio_callback(indata, frames, time, status):
    if status:
        print("Stream status:", status)

    # Grab the first (and only) channel of audio
    frame = indata[:, 0]

    # Print a snippet of the audio data to check if it's meaningful
    print("Audio Frame (first 10 samples):", frame[:10])  # Print the first 10 samples of the frame

    # Compute MFCC
    mfccs = compute_mfcc_cuda(frame, SAMPLE_RATE)

    # Print MFCC
    print("MFCC:", mfccs)

    # Save MFCCs to a text file
    with open(OUTPUT_FILE, "a") as f:
        f.write("MFCC: " + " ".join(map(str, mfccs)) + "\n")


def process_audio_from_file(filename):
    # Load an audio file
    audio_data, samplerate = sf.read(filename)

    # Process in chunks to simulate real-time
    frame_size = 1024
    hop_size = 512
    for i in range(0, len(audio_data), hop_size):
        frame = audio_data[i:i + frame_size]
        mfccs = compute_mfcc_cuda(frame, samplerate)

        # Print MFCC
        print("MFCC:", mfccs)

        # Save MFCCs to a text file
        with open(OUTPUT_FILE, "a") as f:
            f.write("MFCC: " + " ".join(map(str, mfccs)) + "\n")


def main():
    # Ask user whether to use live audio or a file
    user_input = input("Do you want to use live audio (press Enter) or an audio file (enter 'file')? ")

    if user_input.lower() == 'file':
        # Provide the path to the sample audio file
        file_path = input("Please provide the path to the audio file: ")
        process_audio_from_file(file_path)
    else:
        # Open an audio stream to capture audio from the default microphone
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                            blocksize=HOP_SIZE, callback=audio_callback):
            print("Recording... Press Ctrl+C to stop.")
            while True:
                sd.sleep(1000)  # Keep the stream alive


if __name__ == "__main__":
    main()
