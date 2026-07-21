"""Frame sources: webcam, video file, or a single still image looped as a
frame stream. Kept separate from pipeline.py so the GUI and CLI can share one
abstraction regardless of where frames come from.
"""

import cv2


class FrameSource:
    """Wraps cv2.VideoCapture for a camera index or video file."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class StillImageSource:
    """Serves the same still frame every time read() is called.

    Useful for testing/demoing the pipeline without a physical webcam.
    """

    def __init__(self, path: str):
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {path!r}")
        self._frame = frame

    def read(self):
        return self._frame.copy()

    def release(self):
        pass


def open_source(camera=None, video=None, image=None):
    """Resolve CLI args into a frame source. Exactly one of the three should
    be given meaningfully; camera defaults to index 0 if nothing else is set.
    """
    if image:
        return StillImageSource(image)
    if video:
        return FrameSource(video)
    return FrameSource(camera if camera is not None else 0)
