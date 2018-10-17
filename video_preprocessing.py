import cv2
import os

def video_to_frames(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
            cv2.imwrite(out_path + "/frame" + str(frame_count) + ".jpg" , frame)
        else:
            break
    cap.release()


def multi_videos_to_frame(videos_dir, out_path):
    for video in os.listdir(videos_dir):
        video_path = os.path.join(videos_dir, video)
        out_video_path = os.path.join(out_path, video)
        if not os.path.isdir(out_video_path):
            os.mkdir(out_video_path)
        if os.path.isfile(video_path):
            video_to_frames(video_path, out_video_path)


if __name__ == '__main__':
    multi_videos_to_frame("data/Avenue/training_videos", "data/Avenue/training_frames")
