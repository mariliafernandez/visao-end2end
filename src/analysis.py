import cv2
import matplotlib.pyplot as plt


def histogram_equalization(image):
    """Apply CLAHE histogram equalization to an image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(image)
    return result


def histogram(image):
    """Calculate the histogram of an image."""
    return cv2.calcHist([image], [0], None, [256], [0, 256])

def plot_histogram(image, output_path: str):
    """Plot the histogram of an image."""
    hist = histogram(image)
    plt.figure()
    plt.plot(hist, color="k")
    plt.xlim([0, 256])
    plt.title("Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()
