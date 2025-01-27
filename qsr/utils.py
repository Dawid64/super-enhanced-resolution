class SimpleListener:
    def __init__(self, *args, **kwargs):
        pass

    def epoch_callback(self, progress, history):
        pass

    def train_batch_callback(self, progress, history):
        pass

    def val_batch_callback(self, progress, history):
        pass

    def video_loading_callback(self, progress):
        pass
