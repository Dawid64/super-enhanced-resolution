import cv2
import torchvision.transforms as T

class StreamDataset:
    def __init__(self, video_path, original_size=(1920, 1080), target_size=(1280, 720), skip_frames=0):
        self.video_path = video_path
        self.original_size = original_size
        self.target_size = target_size
        self.to_tensor = T.ToTensor()
        self.skip_frames = skip_frames
    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            ret1, frame_fullhd = cap.read()
            if not ret1:
                break
            for _ in range(self.skip_frames):
                cap.grab()
            ret2, frame_hd_gt = cap.read()
            if not ret2:
                break
            for _ in range(self.skip_frames):
                cap.grab()
            frame_fullhd = cv2.resize(frame_fullhd, self.original_size, interpolation=cv2.INTER_AREA)
            frame_fullhd = cv2.cvtColor(frame_fullhd, cv2.COLOR_BGR2RGB)
            frame_hd_gt = cv2.resize(frame_hd_gt, self.original_size, interpolation=cv2.INTER_AREA)
            frame_hd_gt = cv2.cvtColor(frame_hd_gt, cv2.COLOR_BGR2RGB)
            frame_hd = cv2.resize(frame_hd_gt, self.target_size, interpolation=cv2.INTER_AREA)
            yield self.to_tensor(frame_fullhd), self.to_tensor(frame_hd), self.to_tensor(frame_hd_gt)
        cap.release()
