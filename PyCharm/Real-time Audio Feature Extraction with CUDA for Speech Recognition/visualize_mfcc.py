import matplotlib

matplotlib.use('TkAgg')  # Set the backend to TkAgg

import numpy as np
import matplotlib.pyplot as plt


# Function to load the MFCC features from the file
def load_mfcc_features(filename):
    mfcc_features = []

    # Read the MFCC values from the text file
    with open(filename, "r") as file:
        for line in file:
            # Skip lines that don't start with 'MFCC:'
            if line.startswith("MFCC:"):
                # Extract the MFCC values as a list of floats
                mfcc_values = list(map(float, line.strip().split()[1:]))  # Ignore 'MFCC:'
                mfcc_features.append(mfcc_values)

    return np.array(mfcc_features)


# Function to plot MFCCs
def plot_mfcc(mfcc_features):
    # Create a heatmap using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc_features.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='MFCC Value')
    plt.title('MFCC Heatmap')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Frequency Bands')
    plt.tight_layout()
    plt.show()


def main():
    # Load MFCC features from the file and plot them
    mfcc_features = load_mfcc_features("mfcc_features.txt")
    plot_mfcc(mfcc_features)


if __name__ == "__main__":
    main()
