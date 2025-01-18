import cv2
import os
import time

def save_frames_in_time_range(video_path, start_time, end_time):
    video = cv2.VideoCapture(video_path)

    fps = int(video.get(cv2.CAP_PROP_FPS))

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.basename(video_path).split(".")[0]
    os.makedirs(video_name, exist_ok=True)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames

    start_time = time.time()

    frame_number = 1 

    while frame_number <= (end_frame - start_frame + 1):
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_number - 1)
        ret, frame = video.read()

        filename = f"{video_name}/frame_{frame_number:04d}.png"
        cv2.imwrite(filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])

        current_time = time.time()

        remaining_time = (end_frame - start_frame + 1 - frame_number) / fps

        percentage_complete = (frame_number - start_frame + 1) / (end_frame - start_frame + 1) * 100

        print(f"Remaining time: {remaining_time:.2f} seconds\t\tPercentage complete: {percentage_complete:.2f}%")

        frame_number += fps

    video.release()

if __name__ == "__main__":
    video_path = input("Enter Video Path: ")
    start_time = float(input("Enter Start Time (in seconds): "))
    end_time = float(input("Enter End Time (in seconds): "))

    save_frames_in_time_range(video_path, start_time, end_time)

    print("Process Successfully Completed")