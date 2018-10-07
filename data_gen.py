import matplotlib.image as mpimg
import numpy as np
import os


def get_next_batch(data_list, iter, batch_size):
    X_data = []
    index = (iter * batch_size) % len(data_list)
    print(index)
    for i in range(batch_size):
        single_batch = read_one_batch_data(data_list[index + i])
        X_data.append(single_batch)
    return np.array(X_data, dtype=np.float32)


def read_one_batch_data(file_name_list):
    ret = []
    for name in np.nditer(file_name_list): # loop through step num:
        try:
            img = mpimg.imread(str(name)).reshape(-1)
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
            for file in sorted(os.listdir(os.path.join(r, dir))):
                file_path = os.path.join(r, dir, file)
                X_data.append(file_path)
    return np.array(X_data).reshape(int(len(X_data)/step_num), step_num)


def get_data_full(training_dir, step_num):
    X_data = []
    for r, dirs, files in os.walk(training_dir):
        for dir in dirs:
            for file in sorted(os.listdir(os.path.join(r, dir))):
                file_path = os.path.join(r, dir, file)
                try:
                    img = mpimg.imread(str(file_path)).reshape(-1)
                    X_data.append(img)
                except Exception as ex:
                    print("Exception while reading ", file_path, ". Skipping it")
    return np.array(X_data).reshape(int(len(X_data) / step_num), step_num, len(img))


def get_test_data():
    return [], []


def save_images(arr, name):
    folder = os.path.join("data/lstmAE/test", name)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for idx, img in enumerate(arr):
        mpimg.imsave(folder + "/" + str(idx)+".tif", img.reshape(76, 115))