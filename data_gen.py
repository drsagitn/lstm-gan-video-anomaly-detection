import matplotlib.image as mpimg
import numpy as np
import os


def get_next_batch(data_list, index, batch_size):
    X_data = []

    for i in enumerate(batch_size):
        single_batch = read_one_batch_data(data_list[index + i])
        X_data.append(single_batch)

    # for idx, d in enumerate(os.listdir(data_dir)):
    #     if idx >= index and idx < index + batch_size:
    #         X_data.append(read_video(os.path.join(data_dir, d)))
    return np.array(X_data, dtype=np.float32)


def read_one_batch_data(file_name_list):
    ret = []
    for name in file_name_list: # loop through step num:
        try:
            img = mpimg.imread(os.path.join(name, img)).reshape(-1)
            ret.append(img)
        except Exception as ex:
            print("Exception while reading ", img, ". Skipping it")
    return ret


def read_video(video_path):
    ret = []
    for img in os.listdir(video_path):
        try:
            img = mpimg.imread(os.path.join(video_path, img)).reshape(-1)
            ret.append(img)
        except Exception as ex:
            print("Exception while reading ", img, ". Skipping it")
    return ret


def get_train_data(training_dir, step_num):
    X_data = []
    for r, dirs, files in os.walk(training_dir):
        for dir in dirs:
            for file in os.listdir(os.path.join(r, dir)):
                file_path = os.path.join(r, dir, file)
                X_data.append(file_path)
    return np.array(X_data).reshape(int(len(X_data)/step_num), step_num)


def get_test_data():
    return [], []