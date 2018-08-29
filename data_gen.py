import matplotlib.image as mpimg
import numpy as np
import os


def get_next_batch(index, batch_size):
    TRAIN_DIR = 'data/UCSDped1/Train'
    X_data = []
    for idx, d in enumerate(os.listdir(TRAIN_DIR)):
        if idx >= index and idx < index + batch_size:
            X_data.append(read_video(os.path.join(TRAIN_DIR, d)))
    return np.array(X_data, dtype=np.float32)

def read_video(video_path):
    ret = []
    for img in os.listdir(video_path):
        try:
            img = mpimg.imread(os.path.join(video_path, img)).reshape(-1)
            ret.append(img)
        except Exception as ex:
            print("Exception while reading ", img, ". Skipping it")
    return ret