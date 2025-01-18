import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset
import torchvision.transforms as T


class StreamDataset:
    def __init__(self, video_path, original_size=(1920, 1080), target_size=(1280, 720), skip_frames=0, **kwargs):
        self.video_path = video_path
        self.original_size = original_size
        self.target_size = target_size
        self.to_tensor = T.ToTensor()
        self.skip_frames = skip_frames

    def get_video_length(self):
        cap = cv2.VideoCapture(self.video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        i = 0
        while True:
            if not i % self.skip_frames:
                ret1, prev_high_res_frame = cap.read()
                if not ret1:
                    break
                ret2, high_res_frame = cap.read()
                if not ret2:
                    break
                prev_high_res_frame = cv2.resize(prev_high_res_frame, self.original_size, interpolation=cv2.INTER_AREA)
                prev_high_res_frame = cv2.cvtColor(prev_high_res_frame, cv2.COLOR_BGR2RGB)
                high_res_frame = cv2.resize(high_res_frame, self.original_size, interpolation=cv2.INTER_AREA)
                high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
                low_res_frame = cv2.resize(high_res_frame, self.target_size, interpolation=cv2.INTER_AREA)
                yield self.to_tensor(prev_high_res_frame), self.to_tensor(low_res_frame), self.to_tensor(high_res_frame)
            else:
                ret2, high_res_frame = cap.read()
                if not ret2:
                    break
                high_res_frame = cv2.resize(high_res_frame, self.original_size, interpolation=cv2.INTER_AREA)
                high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
                low_res_frame = cv2.resize(high_res_frame, self.target_size, interpolation=cv2.INTER_AREA)
                yield None, self.to_tensor(low_res_frame), self.to_tensor(high_res_frame)
            i += 1
        cap.release()


class NewStreamDataset(IterableDataset):
    """
    A streaming dataset that reads frames from a video file in sequence.
    It generates either a triple (prev_high_res, low_res, high_res) or
    a pair (low_res, high_res) depending on the frame index and skip_frames.

    Args:
        video_file (str): Path to the video file.
        original_size (tuple): Desired high-res frame size, e.g., (1920, 1080).
        target_size (tuple): Desired low-res frame size, e.g., (1280, 720).
        skip_frames (int): Skip interval used in the trainer. If i % skip_frames == 0,
                           the dataset will yield a triple; otherwise a pair.
    """

    def __init__(self, video_file, original_size=(1920, 1080),
                 target_size=(1280, 720), **kwargs):
        super().__init__()
        self.video_file = video_file
        self.original_size = original_size
        self.target_size = target_size

    def get_video_length(self):
        cap = cv2.VideoCapture(self.video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {self.video_file}")
            return
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return

        prev_frame = cv2.resize(prev_frame, self.target_size, interpolation=cv2.INTER_AREA)
        prev_frame = np.transpose(prev_frame, (2, 0, 1)).astype(np.float32) / 255.0

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            high_res_frame = cv2.resize(frame, self.original_size, interpolation=cv2.INTER_AREA)
            high_res_frame = np.transpose(high_res_frame, (2, 0, 1)).astype(np.float32) / 255.0

            low_res_frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            low_res_frame = np.transpose(low_res_frame, (2, 0, 1)).astype(np.float32) / 255.0

            yield (
                torch.from_numpy(prev_frame),
                torch.from_numpy(low_res_frame),
                torch.from_numpy(high_res_frame)
            )

            prev_frame = low_res_frame
            i += 1

        cap.release()
