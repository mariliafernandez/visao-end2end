from src import extract, database
from pathlib import Path

if __name__ == "__main__":

    input_video_path = Path("data/input_video.mp4")
    image_channel_path = input_video_path.parent / "image_channel.mp4"
    audio_channel_path = input_video_path.parent / "audio_channel.wav"

    extract.split_audio_from_video(input_video_path, image_channel_path, audio_channel_path)
    
    time_interval = 0.1  # seconds
    frames, metadata = extract.extract_frames(image_channel_path, time_interval)


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

