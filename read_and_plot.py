from src import database, analysis
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    data = database.load_fits_records()
    # data = data[:10]

    output_histograms_dir = Path("data/histograms")
    output_frames_dir = Path("data/frames")
    output_histograms_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    len_data = len(data)
    for i, record in enumerate(data):
        # save fits data to png
        print(f"Processing {i}/{len_data}")
        image = record['array']
        analysis.plot_histogram(image, output_path=output_histograms_dir / f"histogram_{record['frame_id']:04d}.png")
        image_equalized = analysis.histogram_equalization(image)
        analysis.plot_histogram(image_equalized, output_path=output_histograms_dir / f"histogram_equalized_{record['frame_id']:04d}.png")

        plt.imsave(output_frames_dir / f"frame_{record['frame_id']:04d}.png", image_equalized, cmap='gray')