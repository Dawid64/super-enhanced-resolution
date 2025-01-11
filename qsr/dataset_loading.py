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
            ret1, prev_high_res_frame = cap.read()
            if not ret1:
                break
            for _ in range(self.skip_frames):
                cap.grab()
            ret2, high_res_frame = cap.read()
            if not ret2:
                break
            for _ in range(self.skip_frames):
                cap.grab()
            prev_high_res_frame = cv2.resize(
                prev_high_res_frame, self.original_size, interpolation=cv2.INTER_AREA)
            prev_high_res_frame = cv2.cvtColor(
                prev_high_res_frame, cv2.COLOR_BGR2RGB)
            high_res_frame = cv2.resize(
                high_res_frame, self.original_size, interpolation=cv2.INTER_AREA)
            high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
            low_res_frame = cv2.resize(
                high_res_frame, self.target_size, interpolation=cv2.INTER_AREA)
            yield self.to_tensor(prev_high_res_frame), self.to_tensor(low_res_frame), self.to_tensor(high_res_frame)
        cap.release()
