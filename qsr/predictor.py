from typing import Dict
import cv2
import numpy as np
import torch
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from .model import SrCNN2

from qsr.utils import SimpleListener

from .dataset_loading import NewStreamDataset


class Upscaler:
    def __init__(self, model_path, device='auto', original_size=(1920, 1080), target_size=(1280, 720), listener=None):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.original_size = original_size
        self.target_size = target_size
        upscale_factor = original_size[0] / target_size[0]
        model = SrCNN2(upscale_factor=upscale_factor)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = model.to(device)
        self.dataset_format = NewStreamDataset
        self.listener: SimpleListener = listener
        self.run_name = "QSR_Upscaling"
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def _log_params(self, parameters: Dict):
        for key, value in parameters.items():
            mlflow.log_param(key, value)

    def upscale(self, video_file, num_frames=-1, skip_frames=10, fps=60.0, video_path_out="output.mp4"):
        writer = cv2.VideoWriter(video_path_out, self.fourcc, fps, self.original_size)

        dataset = self.dataset_format(video_file=video_file, original_size=self.original_size, target_size=self.target_size)
        if num_frames == -1:
            num_frames = dataset.get_video_length()-1
        self.model.eval()
        mlflow.set_experiment("quantum_super_resolution_experiment")
        with mlflow.start_run(run_name=self.run_name):
            self._log_params({"video_file": video_file, "input_res": self.target_size, "output_res": self.original_size,
                             "num_frames": num_frames, "skip_frames": skip_frames, "fps": fps})
        pbar = tqdm(dataset, total=num_frames, unit='frames')
        for i, frames in enumerate(pbar):
            with torch.no_grad():
                prev_frame, frame, _ = [f.unsqueeze(0).to(self.device) for f in frames]
                pred_frame = self.model(prev_frame, frame)
                pbar.set_postfix({'frame': i})
                if self.listener is not None:
                    self.listener.callback(frame=i/num_frames)
                out = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out, 0, 1)
                out = (out * 255).astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                writer.write(out)
        cv2.destroyAllWindows()
        writer.release()
