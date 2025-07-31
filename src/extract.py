import cv2
from astropy.io import fits
from moviepy import VideoFileClip
from pathlib import Path


def split_audio_from_video(input_video_path: Path, image_channel_path: Path, audio_channel_path: Path):
    video = VideoFileClip(input_video_path)
    video.audio.write_audiofile(audio_channel_path, codec="pcm_s32le")
    video.write_videofile(image_channel_path, audio=False)
    video.close()


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to a timestamp string in HH:MM:SS.sss format."""
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def extract_frames(video_path: Path, time_interval: float) -> tuple[list, list]:
    """
    Extract frames from a video file at specified time intervals.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames = []
    metadata = []
    current_time = 0

    while current_time <= duration:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        metadata.append(
            {"frame_id": frame_idx, "timestamp": seconds_to_timestamp(current_time)}
        )
        current_time += time_interval
    cap.release()
    return frames, metadata


def image_to_fits(img: str, metadata: dict):
    """Convert an image to FITS format"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hdu = fits.PrimaryHDU(gray_img)
    hdu.header["TIMESTAMP"] = metadata["timestamp"]
    hdu.header["FRAME_ID"] = metadata["frame_id"]
    return hdu

