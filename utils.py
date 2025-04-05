import cv2
import os
import numpy as np
from collections import defaultdict


def frames_to_video(frames_folder, output_path, fps=30):
    # Example usage
    # frames_folder = 'VisDrone/VisDrone2019-MOT-train/sequences/uav0000020_00406_v'  # Replace with the folder containing your frames
    # output_path = 'output_video.mp4'
    # frames_to_video(frames_folder, output_path, fps=30)
    
    # Get list of files and sort them (assuming frame filenames are sortable in order)
    """
    Convert a folder of frames to a video.

    Parameters
    ----------
    frames_folder : str
        Path to the folder containing the frames.
    output_path : str
        Path to save the video file.
    fps : int, optional
        Frames per second for the output video. Default is 30.

    Notes
    -----
    Frames are sorted alphabetically in the folder, so make sure the frame
    filenames are sortable in order.
    """
    frame_files = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not frame_files:
        print("No image files found in the folder.")
        return

    # Read the first frame to get video dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame is not None:
            out.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_file}")

    out.release()
    print(f"Video saved to {output_path}")




def parse_labels(label_path):
    """
    Parses label file into a dictionary mapping frame numbers to list of detections.
    Each detection is a tuple: (id, left, top, width, height)
    """
    tracks = defaultdict(list)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            obj_id = int(parts[1])
            bb_left = int(float(parts[2]))
            bb_top = int(float(parts[3]))
            bb_width = int(float(parts[4]))
            bb_height = int(float(parts[5]))
            tracks[frame].append((obj_id, bb_left, bb_top, bb_width, bb_height))
    return tracks

def visualize_tracking(frames_dir, label_path, output_path='output_tracking.mp4', fps=30):
    # visualize_tracking('/home/khanh/data/LPT/data/MOT/MOT17/train/MOT17-02-DPM/img1', '/home/khanh/data/LPT/data/MOT/MOT17/train/MOT17-02-DPM/gt/gt.txt')

    """
    Visualizes tracking results by drawing bounding boxes and track paths onto a video.

    Parameters
    ----------
    frames_dir : str
        Path to the folder containing the frames of the video (jpg format).
    label_path : str
        Path to the MOTChallenge label file.
    output_path : str, optional
        Path to the output video file (mp4 format). Default is 'output_tracking.mp4'.
    fps : int, optional
        Frames per second for the output video. Default is 30.

    Notes
    -----
    Frames are sorted alphabetically in the folder, so make sure the frame
    filenames are sortable in order.
    """
    tracks = parse_labels(label_path)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]

    # Create output video writer
    if not frame_paths:
        raise ValueError("No frames found.")
    sample_frame = cv2.imread(frame_paths[0])
    height, width = sample_frame.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Store track paths
    track_paths = defaultdict(list)  # id -> list of (center_x, center_y)

    for idx, frame_path in enumerate(frame_paths, 1):
        frame = cv2.imread(frame_path)
        detections = tracks.get(idx, [])

        for obj_id, x, y, w, h in detections:
            color = (37 * obj_id % 255, 17 * obj_id % 255, 29 * obj_id % 255)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            center = (x + w // 2, y + h // 2)

            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, f'ID {obj_id}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Add current center to track path
            track_paths[obj_id].append(center)

            # Draw track path
            path_points = track_paths[obj_id]
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i - 1], path_points[i], color, 2)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")