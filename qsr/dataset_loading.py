from typing import Literal
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import ffmpegcv
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from time import sleep


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
        pbar = tqdm(self.video_paths, desc="Loading videos", leave=False)
        for i, video_path in enumerate(pbar):
            self.load_video(frames_backward, frames_forward, video_path)
            sleep(5)  # helps a little to ease the peak system memory usage
            if listener:
                listener.video_loading_callback((i+1)/len(self.video_paths))

    def load_video(self, frames_backward, frames_forward, video_path):
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

    def __len__(self):
        return len(self.frame_windows)

    def __getitem__(self, idx):
        frames = self.frame_windows[idx]
        prev_frames = list(frames[:self.frames_backward])
        cur_frame = frames[self.frames_backward]
        next_frames = list(frames[self.frames_backward + 1:])
        for i in range(self.frames_backward):
            size = prev_frames[i].shape[0]
            if size > self.target_size[1]:
                prev_frames[i] = cv2.resize(prev_frames[i], self.target_size, interpolation=cv2.INTER_AREA)
            elif size == self.target_size[1]:
                pass
            else:
                raise Exception("Video cannot be smaller than the low resolution size")
            prev_frames[i] = torch.from_numpy(np.transpose(prev_frames[i], (2, 0, 1))).float() / 255.0
        prev_frames = torch.concat(prev_frames, dim=0)
        size = cur_frame.shape[0]
        if self.mode == 'training':
            if size > self.original_size[1]:
                high_res_frame = cv2.resize(cur_frame, self.original_size, interpolation=cv2.INTER_AREA)
            elif size == self.original_size[1]:
                high_res_frame = cur_frame
            else:
                raise Exception("Video cannot be smaller than the high resolution except for the inference mode")
            high_res_frame = torch.from_numpy(np.transpose(high_res_frame, (2, 0, 1))).float() / 255.0
        if size > self.target_size[1]:
            low_res_frame = cv2.resize(cur_frame, self.target_size, interpolation=cv2.INTER_AREA)
        elif size == self.target_size[1]:
            low_res_frame = cur_frame
        else:
            raise Exception("Video cannot be smaller than the low resolution size")
        low_res_frame = torch.from_numpy(np.transpose(low_res_frame, (2, 0, 1))).float() / 255.0
        for i in range(self.frames_forward):
            size = next_frames[i].shape[0]
            if size > self.target_size[1]:
                next_frames[i] = cv2.resize(next_frames[i], self.target_size, interpolation=cv2.INTER_AREA)
            elif size == self.target_size[1]:
                pass
            else:
                raise Exception("Video cannot be smaller than the low resolution size")
            next_frames[i] = torch.from_numpy(np.transpose(next_frames[i], (2, 0, 1))).float() / 255.0
        next_frames = torch.concat(next_frames, dim=0)
        return ((prev_frames, low_res_frame, next_frames),  high_res_frame) if self.mode == 'training' else (prev_frames, low_res_frame, next_frames)


if __name__ == "__main__":
    dataset = MultiVideoDataset(["videos/HD/1.mp4", "videos/HD/2.mp4"])
