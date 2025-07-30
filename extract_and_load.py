from src import extract, database

if __name__ == "__main__":
    video_path = "data/image_channel.mp4"
    time_interval = 0.1  # seconds
    frames, metadata = extract.extract_frames(video_path, time_interval)

    for i, (frame, meta) in enumerate(zip(frames, metadata)):
        hdu = extract.image_to_fits(frame, meta)
        print(f"Saving frame {i} to FITS format with metadata: {meta}")
        database.insert_fits_data({
            "fits_data": hdu.data.tobytes(),
            "timestamp": hdu.header["TIMESTAMP"],
            "frame_id": hdu.header.get("FRAME_ID"),
            'width': hdu.data.shape[1],
            'height': hdu.data.shape[0],
            'dtype': str(hdu.data.dtype),
        })
