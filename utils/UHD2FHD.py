import cv2
import glob
from tqdm import tqdm


def resize_videos(input_dir: str = 'videos/UHD', output_dir: str = 'videos/FHD', output_resolution: tuple = (1920, 1080)):

    print(len(glob.glob(f"{input_dir}/*.mp4")))
    progress_bar = tqdm(glob.glob(f"{input_dir}/*.mp4"), unit='video')

    for video_path in progress_bar:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        output_path = f"{output_dir}/{video_path.split('/')[-1]}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 25.0, output_resolution)
        while True:
            ret1, frame_before = cap.read()
            if not ret1:
                break
            frame_after = cv2.resize(frame_before, output_resolution, interpolation=cv2.INTER_AREA)
            writer.write(frame_after)
        cap.release()
        writer.release()


resize_videos()
