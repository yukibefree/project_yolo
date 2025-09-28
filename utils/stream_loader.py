import cv2
import math
import os
import time
import urllib
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER

# utils.py から必要なヘルパー関数をインポート
from .utils import check_requirements, clean_str, IS_COLAB, IS_KAGGLE

@dataclass
class SourceTypes:
    stream: bool = False
    screenshot: bool = False
    from_img: bool = False
    tensor: bool = False

def get_best_youtube_url(url: str, method: str = "yt-dlp") -> str | None:
    # utils.pyのcheck_requirementsを呼び出す
    if method == "pytube":
        check_requirements("pytubefix>=6.5.2")
        from pytubefix import YouTube
        streams = YouTube(url).streams.filter(file_extension="mp4", only_video=True)
        streams = sorted(streams, key=lambda s: s.resolution, reverse=True)
        for stream in streams:
            if stream.resolution and int(stream.resolution[:-1]) >= 1080:
                return stream.url
    elif method == "pafy":
        check_requirements(("pafy", "youtube_dl==2020.12.2"))
        import pafy
        return pafy.new(url).getbestvideo(preftype="mp4").url
    elif method == "yt-dlp":
        check_requirements("yt-dlp")
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
        for f in reversed(info_dict.get("formats", [])):
            good_size = (f.get("width") or 0) >= 1920 or (f.get("height") or 0) >= 1080
            if good_size and f["vcodec"] != "none" and f["acodec"] == "none" and f["ext"] == "mp4":
                return f.get("url")
    return None
  
class LoadStreams:
    def __init__(self, sources: str = "file.streams", vid_stride: int = 1, buffer: bool = False, channels: int = 3):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.count = 0
        self.buffer = buffer  # buffer input streams
        self.running = True  # running flag for Thread
        self.mode = "stream"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR  # grayscale or RGB

        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        n = len(sources)
        self.bs = n
        self.fps = [0] * n  # frames per second
        self.frames = [0] * n
        self.threads = [None] * n
        self.caps = [None] * n  # video capture objects
        self.imgs = [[] for _ in range(n)]  # images
        self.shape = [[] for _ in range(n)]  # image shapes
        self.sources = [clean_str(x).replace(os.sep, "_") for x in sources]  
        for i, s_original in enumerate(sources):
            s = s_original
            if urllib.parse.urlparse(s).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
                s = get_best_youtube_url(s)
                if s is None:
                    raise ConnectionError(f"Failed to get YouTube stream URL from {s_original}")
            if s == 0 and (IS_COLAB or IS_KAGGLE):
                raise NotImplementedError(
                    "'source=0' webcam not supported in Colab and Kaggle notebooks. "
                    "Try running 'source=0' in a local environment."
                )
            self.caps[i] = cv2.VideoCapture(s)  # store video capture object
            if not self.caps[i].isOpened():
                raise ConnectionError(f"{s}Failed to open {s}")
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.caps[i].get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames[i] = max(int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float(
                "inf"
            )  # infinite stream fallback
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            success, im = self.caps[i].read()  # guarantee first frame
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
            if not success or im is None:
                raise ConnectionError(f"{s}Failed to read images from {s}")
            self.imgs[i].append(im)
            self.shape[i] = im.shape
            self.threads[i] = Thread(target=self.update, args=([i, self.caps[i], s]), daemon=True)
            LOGGER.info(f"{s}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info("")  # newline

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        """Read stream frames in daemon thread and update image buffer."""
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and cap.isOpened() and n < (f - 1):
            if len(self.imgs[i]) < 30:  # keep a <=30-image buffer
                n += 1
                cap.grab()  # .read() = .grab() followed by .retrieve()
                if n % self.vid_stride == 0:
                    success, im = cap.retrieve()
                    im = (
                        cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None] if self.cv2_flag == cv2.IMREAD_GRAYSCALE else im
                    )
                    if not success:
                        im = np.zeros(self.shape[i], dtype=np.uint8)
                        LOGGER.warning("Video stream unresponsive, please check your IP camera connection.")
                        cap.open(stream)  # re-open stream if signal was lost
                    if self.buffer:
                        self.imgs[i].append(im)
                    else:
                        self.imgs[i] = [im]
            else:
                time.sleep(0.01)  # wait until the buffer is empty

    def close(self):
        """Terminate stream loader, stop threads, and release video capture resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for cap in self.caps:  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                LOGGER.warning(f"Could not release VideoCapture object: {e}")

    def __iter__(self):
        """Iterate through YOLO image feed and re-open unresponsive streams."""
        self.count = -1
        return self

    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        """Return the next batch of frames from multiple video streams for processing."""
        self.count += 1

        images = []
        for i, x in enumerate(self.imgs):
            # Wait until a frame is available in each buffer
            while not x:
                if not self.threads[i].is_alive():
                    self.close()
                    raise StopIteration
                time.sleep(1 / min(self.fps))
                x = self.imgs[i]
                if not x:
                    LOGGER.warning(f"Waiting for stream {i}")

            # Get and remove the first frame from imgs buffer
            if self.buffer:
                images.append(x.pop(0))

            # Get the last frame, and clear the rest from the imgs buffer
            else:
                images.append(x.pop(-1) if x else np.zeros(self.shape[i], dtype=np.uint8))
                x.clear()

        return self.sources, images, [""] * self.bs

    def __len__(self) -> int:
        """Return the number of video streams in the LoadStreams object."""
        return self.bs  # 1E12 frames = 32 streams at 30 FPS for 30 years