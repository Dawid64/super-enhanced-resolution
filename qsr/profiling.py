import cProfile
import pstats
from .trainer import Trainer

def profile():
    with cProfile.Profile() as pr:
        trainer = Trainer()
        trainer.train_model(num_epochs=2, video_file='Inter4k/60fps/small/1.mp4')
    stats = pstats.Stats(pr)
    stats.dump_stats("profile.stats")
    
if __name__ == '__main__':
    profile()