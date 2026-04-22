from __future__ import annotations

import cv2


class CameraStream:
    def __init__(self, camera_index: int = 0, mirror: bool = True) -> None:
        self.camera_index = int(camera_index)
        self.mirror = bool(mirror)
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")

    def read(self):
        success, frame = self.capture.read()
        if success and self.mirror:
            frame = cv2.flip(frame, 1)
        return success, frame

    def show(self, window_name: str, frame) -> None:
        cv2.imshow(window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def close(self) -> None:
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
