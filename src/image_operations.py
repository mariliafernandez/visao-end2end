import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    """
    Apply CLAHE histogram equalization to image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(image)
    return result


def histogram(image):
    """
    Calculate image histogram
    """
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def plot_histogram(image, output_path: str):
    """
    Plot image histogram and save to output path
    """
    hist = histogram(image)
    plt.figure()
    plt.plot(hist, color="k")
    plt.xlim([0, 256])
    plt.title("Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


def extract_histogram_features(img: np.ndarray) -> list:
    """
    Extract histogram features from an image
    """
    hist = histogram(img)
    hist = cv2.normalize(hist, hist).flatten()  # Normaliza o histograma
    total = np.sum(hist)
    values = np.arange(256)

    mean = np.sum(hist * values) / total
    std = np.sqrt(np.sum(hist * (values - mean) ** 2) / total)
    skewness = np.sum(hist * (values - mean) ** 3) / (total * std**3 + 1e-8)
    kurtosis = np.sum(hist * (values - mean) ** 4) / (total * std**4 + 1e-8)
    entropy = -np.sum((hist / total) * np.log2(hist / total + 1e-8))

    # Percentis
    cdf = np.cumsum(hist) / total
    p25 = np.searchsorted(cdf, 0.25)
    p50 = np.searchsorted(cdf, 0.50)  # Mediana
    p75 = np.searchsorted(cdf, 0.75)

    return [
        mean,
        std,
        skewness,
        kurtosis,
        entropy,
        p25,
        p50,
        p75,
    ]  # len(histogram_features) = 8


def extract_texture(
    img: np.ndarray,
) -> list:
    """
    Extract texture features from an image using Gabor filters
    """
    ksize = 31
    sigma = 4.0
    lambdas = [9, 10]
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gamma = 0.5
    psi = 0
    texture_features = []

    for lamda in lambdas:  # 2 lambdas
        for theta in thetas:  # 4 thetas
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            mean = filtered.mean()
            std = filtered.std()
            energy = np.sum(filtered**2) / filtered.size

            texture_features.extend([mean, std, energy])  # 3 features por filtro
    return texture_features  # len(texture_features) = 24 (2 * 4 * 3)


def extract_corners(img: np.ndarray) -> list:
    """
    Extract corner features from an image using Shi-Tomasi corner detection
    """
    corners = cv2.goodFeaturesToTrack(
        img, maxCorners=200, qualityLevel=0.01, minDistance=10
    )
    if corners is not None:
        num_corners = len(corners)
        x_mean = np.mean(corners[:, 0, 0])
        y_mean = np.mean(corners[:, 0, 1])
        x_std = np.std(corners[:, 0, 0])
        y_std = np.std(corners[:, 0, 1])
    else:
        # Se não detectou nenhum canto
        num_corners = 0
        x_mean = y_mean = x_std = y_std = 0

    corner_features = [num_corners, x_mean, y_mean, x_std, y_std]
    return corner_features  # len(corner_features) = 5


def extract_edges(img: np.ndarray) -> list:
    """
    Extract edge features from an image using Canny edge detection
    """
    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    edge_density = np.sum(edges > 0) / edges.size

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # gradientes em x
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # gradientes em y

    gradient_orientation = np.arctan2(sobely, sobelx)
    mask = edges > 0

    if np.any(mask):
        mean_orientation = np.mean(gradient_orientation[mask])
        std_orientation = np.std(gradient_orientation[mask])
    else:
        mean_orientation = std_orientation = 0

    edge_features = [edge_density, mean_orientation, std_orientation]
    return edge_features  # len(edge_features) == 3


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from an image
    """
    features = []

    # Extrai textura
    # texture_features = extract_texture(img)
    # features.extend(texture_features)  # len(texture_features) = 24

    # Extrai cantos
    corner_features = extract_corners(img)
    features.extend(corner_features)  # len(corner_features) = 5

    # Extrai bordas
    edge_features = extract_edges(img)
    features.extend(edge_features)  # len(edge_features) = 3

    # Extrai histograma
    histogram_features = extract_histogram_features(img)
    features.extend(histogram_features)  # len(histogram_features) = 8

    return np.asarray(features)


