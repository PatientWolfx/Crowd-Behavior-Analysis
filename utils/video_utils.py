import cv2
from backend.inference import detect_people
from backend.behavior import crowd_density, detect_behavior
from utils.motion_utils import compute_motion
from utils.heatmap_utils import generate_heatmap
from utils.alert_utils import generate_alert

import numpy as np

def process_multiple_videos(video_paths):

    caps = []
    prev_frames = []

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        caps.append(cap)
        prev_frames.append(None)

    while True:
    
        frames = []
    
        for i, cap in enumerate(caps):
    
            ret, frame = cap.read()
    
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
    
            frame = cv2.resize(frame,(400,300))
    
            people = detect_people(frame)
    
            count, density_status = crowd_density(people)
    
            motion = 0
    
            if prev_frames[i] is not None:
                motion = compute_motion(prev_frames[i], frame)
    
            behavior = detect_behavior(count, motion)
    
            alert = generate_alert(behavior, i+1)
    
            frame = generate_heatmap(frame, people)
    
            cv2.putText(frame, f"Camera {i+1}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
            cv2.putText(frame, f"People: {count}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
            cv2.putText(frame, f"Behavior: {behavior}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    
            if alert:
                print(alert)
    
                cv2.putText(frame,
                            alert,
                            (10,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2)
    
            frames.append(frame)
    
            prev_frames[i] = frame.copy()
    
        if len(frames) == 0:
            continue
    
        dashboard = create_camera_grid(frames)
    
        cv2.imshow("Multi-Camera Crowd Monitoring", dashboard)
    
        if cv2.waitKey(1) & 0xFF == 27:
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()

def create_camera_grid(frames):

    if len(frames) == 4:

        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))

        grid = np.vstack((top_row, bottom_row))

        return grid

    elif len(frames) == 2:

        grid = np.hstack((frames[0], frames[1]))

        return grid

    else:
        return frames[0]
