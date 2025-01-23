from typing import Literal
import cv2
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader, random_split
import torchvision.transforms as T
import ffmpegcv
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


class MultiVideoDataset(Dataset):
    def __init__(self, video_paths: list[str], original_size=(1920, 1080), target_size=(1280, 720), frames_backward=2, frames_forward=2, listener=None, mode: Literal['training', 'inference'] = 'training'):
        super().__init__()
        self.video_paths = video_paths
        self.original_size = original_size
        self.target_size = target_size
        self.frames_backward = frames_backward
        self.frames_forward = frames_forward
        self.frame_windows = np.array([])
        self.mode = mode
        pbar = tqdm(self.video_paths, desc="Loading videos")
        for i, video_path in enumerate(pbar):
            if listener:
                listener.video_loading_callback((i+1)/len(self.video_paths))
            frames = []
            reader = ffmpegcv.VideoCaptureNV(video_path, pix_fmt='rgb24') if torch.cuda.is_available() else ffmpegcv.VideoCapture(video_path, pix_fmt='rgb24')
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                frames.append(frame)
            reader.release()
            frames = np.array(frames)
            windows = sliding_window_view(frames, (frames_backward + frames_forward + 1, *frames[0].shape)).squeeze().astype(np.uint8)
            self.frame_windows = np.concatenate([self.frame_windows, windows]) if self.frame_windows.size else windows
        print(self.frame_windows.nbytes / 1024**3, "GB")

    def __len__(self):
        return len(self.frame_windows)

    def __getitem__(self, idx):
        frames = self.frame_windows[idx]
        prev_frames = list(frames[:self.frames_backward])
        cur_frame = frames[self.frames_backward]
        next_frames = list(frames[self.frames_backward + 1:])
        for i in range(self.frames_backward):
            prev_frames[i] = cv2.resize(prev_frames[i], self.target_size, interpolation=cv2.INTER_AREA)
            prev_frames[i] = torch.from_numpy(np.transpose(prev_frames[i], (2, 0, 1))).float() / 255.0
        prev_frames = torch.concat(prev_frames, dim=0)
        if self.mode == 'training':
            high_res_frame = cv2.resize(cur_frame, self.original_size, interpolation=cv2.INTER_AREA)
            high_res_frame = torch.from_numpy(np.transpose(high_res_frame, (2, 0, 1))).float() / 255.0
        low_res_frame = cv2.resize(cur_frame, self.target_size, interpolation=cv2.INTER_AREA)
        low_res_frame = torch.from_numpy(np.transpose(low_res_frame, (2, 0, 1))).float() / 255.0
        for i in range(self.frames_forward):
            next_frames[i] = cv2.resize(next_frames[i], self.target_size, interpolation=cv2.INTER_AREA)
            next_frames[i] = torch.from_numpy(np.transpose(next_frames[i], (2, 0, 1))).float() / 255.0
        next_frames = torch.concat(next_frames, dim=0)
        return ((prev_frames, low_res_frame, next_frames),  high_res_frame) if self.mode == 'training' else (prev_frames, low_res_frame, next_frames)


if __name__ == "__main__":
    dataset = MultiVideoDataset(["videos/HD/1.mp4", "videos/HD/2.mp4"])


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
        cap = ffmpegcv.VideoCaptureNV(self.video_file, pix_fmt='rgb24') if torch.cuda.is_available(
        ) else ffmpegcv.VideoCapture(self.video_file, pix_fmt='rgb24')
        length = len(cap)
        cap.release()
        return length

    def __iter__(self):
        cap = ffmpegcv.VideoCaptureNV(self.video_file, pix_fmt='rgb24') if torch.cuda.is_available(
        ) else ffmpegcv.VideoCapture(self.video_file, pix_fmt='rgb24')
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
